import sys
sys.path.append(".")

from comet_ml import Experiment
import torch
import string, random
import numpy as np
from utils.rs import ssim
import traceback


import gc

from exploration import get_model_dataset, back_transform
from attacks import prepare_message
from attacks.art import generate as generate_art
from attacks.torchattacks import generate as generate_ta

from utils import init_comet
from attacks import success_strategy

def run(parameters, name="steganography_PGD", random_seed=0, experiment=None, return_model=False, model=None,
        batch_t=None, batch_index=0, save_image_each=100 ):
    random.seed(random_seed)
    print("running", parameters)
    batch, m = get_model_dataset(parameters, device=parameters.get("device","cuda"))

    batch_t = batch_t if batch_t is not None else batch
    model = model if model is not None else m

    if parameters.get("device","cuda") =="cuda":
        batch_t = batch_t.cuda()

    with torch.no_grad():
        out = model(batch_t)

    if parameters.get("model") in ["coco_v3","coco_v2","voc2012_","coco_yolov5"]:
        parameters["dense"] = True
        parameters["output_size"] = out.shape[-1]
    else:
        parameters["dense"] = False

    if parameters.get("use_hidden"):
        min_val,max_val = out.min().item(),out.max().item()
    else:
        min_val, max_val = 0, 1

    padding = False
    capacity = out.shape[1]
    print("capacity", capacity)

    if not parameters.get("binary_message",False):
        letters = string.ascii_letters

        message = parameters.get("message", None)

        if parameters.get("message", None) is None:
            message = ''.join(random.choice(letters) for i in range(int(batch_t.shape[2] * batch_t.shape[3] / 7 *
                                                                    parameters.get("bpp",1))))

        encoded, absolute_capacity, density = prepare_message(message, capacity=capacity,
                                                              input_shape=batch_t[0].shape,
                                                              min_val=min_val,
                                                              max_val=max_val, padding=padding,
                                                              parameters=parameters)

    elif parameters.get("binary_message",False)=="flip_target":
        classes = torch.argmax(out,1)
        encoded = np.ones(out.shape)
        for i, a in enumerate(classes):
            encoded[i, a] = 0
        #encoded = ["{}".format("".join(o)) for o in out]
        absolute_capacity = out.shape[1]
        density = absolute_capacity / (batch_t.shape[2] * batch_t.shape[3])



    if encoded is None:
        return

    parameters["model_capacity"] = capacity
    parameters["message_size"] = absolute_capacity
    parameters["density"] = density
    parameters["decision_threshold"] = (min_val + max_val) / 2
    data_size = absolute_capacity if parameters.get("strategy") != "sorted" else capacity

    if experiment is None:
        experiment = init_comet(args= parameters, project_name=name)

    min_pixel_value, max_pixel_value = batch_t.min(), batch_t.max() #0, 1

    max_pixel_value = 0.5

    x_test = batch_t.float()
    y_test = model(x_test)

    if len(encoded.shape)<2:
        y_target = torch.from_numpy(encoded).unsqueeze(0).type(torch.float).to(batch_t.get_device())
        y_target = y_target.repeat(x_test.shape[0], 1)

        if not padding and parameters.get("strategy") != "sorted":
            y_target = torch.cat([y_target, y_test.detach()[:, absolute_capacity:]], dim=1)
    else:
        y_target = torch.from_numpy(encoded).type(torch.float).to(batch_t.get_device())

    if parameters.get("lib") =="art":
        x_test_adv = generate_art(parameters, model,x_test,y_target , min_pixel_value, max_pixel_value, min_val, max_val,
                                  capacity, data_size=data_size, strategy=parameters.get("strategy"))


    elif parameters.get("lib") =="torchattack":
        x_test_adv = generate_ta(parameters, model,x_test,y_target , min_pixel_value, max_pixel_value, min_val, max_val,
                                 capacity, experiment=experiment, data_size=data_size, strategy=parameters.get("strategy"), batch_index=batch_index)


    if isinstance(x_test_adv,dict):
        best_adv_images = x_test_adv.get("x_test_adv")
        best_adv_score = x_test_adv.get("best_adv_score")

        mean_adv_images = x_test_adv.get("mean_adv_images")
        mean_adv_score = x_test_adv.get("mean_adv_score")

        max_adv_images = x_test_adv.get("max_adv_images")
        max_adv_score = x_test_adv.get("max_adv_score")

        x_test_adv = mean_adv_images.cuda()

    if isinstance(x_test_adv, np.ndarray):
        x_test_adv = torch.from_numpy(x_test_adv).cuda()

    with torch.no_grad():
        predictions = model(x_test_adv)

    p = predictions.cpu().detach().numpy()>(min_val+max_val)/2 if parameters.get("strategy") == "binary" else predictions.cpu().detach().numpy()
    y = y_target.cpu().detach().numpy()>(min_val+max_val)/2 if parameters.get("strategy") == "binary" else y_target.cpu().detach().numpy()
    success = p == y
    success = success.sum(axis=1)/success.shape[1] #success.sum() / success.size

    success1 = success_strategy(p,y,strategy=parameters.get("strategy"),data_size=absolute_capacity)

    sim = [ssim(torch.FloatTensor(img.cpu()).permute(2, 1, 0).unsqueeze(0),torch.FloatTensor(x_test_adv[i].cpu())
               .permute(2, 1, 0).unsqueeze(0)).item() for (i,img) in enumerate(batch_t)]
    print("success strategy adv",success1, "ssim",sim)

    #transformed_clean = back_transform(x_test, parameters.get("dataset"))
    transformed_clean = x_test
    #transformed_adv = back_transform(x_test_adv, parameters.get("dataset"))
    transformed_adv =x_test_adv

    for (i,s) in enumerate(success):
        index = i+len(success)*batch_index
        experiment.log_metric("accuracy", success[i], step=index )
        experiment.log_metric("accuracy_data", success1[i], step=index)
        experiment.log_metric("rs-bpp_data", (2 * success1[i] - 1) * density, step=index)
        experiment.log_metric("ssim",sim[i],step=i,epoch=index)


        if batch_index==0 or index%save_image_each==0:
            experiment.log_image(transformed_clean[i].cpu(), name='Val/image clean ', image_channels='first', step=index)
            experiment.log_image(transformed_adv[i].cpu(), name='Val/image adv ', image_channels='first', step=index)

    if return_model:
        return experiment, success1, x_test_adv, x_test, model, y_target , parameters

    else:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return experiment, success1, x_test_adv, x_test


if __name__ == '__main__':

    criteria = ["mse","ce","kl","hl"]
    criteria = ["mse"]
    hidden_layers = [True,False]
    hidden_layers = [True]
    algorithms=["art_pgd","art_mpgd","art_ipgd","art_apgd","art_bound","torchattack_mifgsm","torchattack_pixel"]
    algorithms = ["torchattack_mifgsm"]
    densities = [0.04, 0.1, 0.4, 1, 4]
    densities = [0.3, 1,4]
    models = ["imagenet_resnet18","cifar10_resnet20", "cifar10_resnet56","imagenet_wide_resnet50"]
    models = ["cifar10_resnet20", "imagenet_resnet18"]
    datasets = ["div2k"]
    max_eps = [0.2,0.5,1,2,4,8]
    max_eps = [0.2, 0.5, 1]

    parameters = {"use_hidden" : True, "pretrained" : True, "criterion" : "mse_vSLS", "algorithm":"mifgsm", "max_eps":10,
                  "norm":"Linf", "max_iter":1000, "eps_step":0.05,"num_random_init":10, "batch_size":2 ,
                  "success_threshold": 0.9, "bpp":1, "lib":"torchattack", "model" : "cifar10_resnet20",
                  "dataset" : "cifar10", "strategy":"sorted", "random_noise":"fading", "reduction":"mean"}

    #parameters = {"use_hidden": False, "pretrained": True, "criterion": "mse", "algorithm": "mifgsm", "max_eps": 0.5,
    #              "norm": "Linf", "max_iter": 50000, "eps_step": 0.05, "num_random_init": 5, "batch_size": 2,
    #              "success_threshold": 0.9, "bpp": 0.3, "lib": "torchattack", "model": "imagenet_resnet18",
    #              "dataset": "div2k"}

    run(parameters,name="sorted_loss")
    exit()

    parameters["batch_size"] = 1000
    parameters["strategy"] = "binary"
    parameters["max_iter"] = 100
    #parameters["algorithm"]="pgd"
    #parameters["max_eps"] = 32
    #parameters["lib"]="art"

    for c in criteria:
        parameters["criterion"] = c
        for h in hidden_layers:
            parameters["use_hidden"] = h
            for al in algorithms:
                library, algorithm = al.split("_")
                parameters["lib"] = library
                parameters["algorithm"] = algorithm
                for den in densities:
                    parameters["bpp"] = den
                    for m in models:
                        parameters["model"] = m
                        for dts in datasets:
                            parameters["dataset"] = dts
                            for eps in max_eps:
                                parameters["max_eps"] = eps
                                try:
                                    run(parameters,name="stegano_MIM_Large_random")
                                except Exception as e:
                                    print(e)
                                    traceback.print_exc()

"""
model.classifier = nn.Sequential(*[model.classifier[i] for i in range(2)])
print(model.classifier)
print("all")

CUDA_VISIBLE_DEVICES=2 /home/sghamizi/miniconda3/envs/steganogan/bin/python ./exploration/torch_layers.py
"""

#tested: imagenet like cover (3x224x224); pretrained resnet18 with imagenet dataset 500 steps; 5 random init
# last layer (1000): MSE / CrossEntropy / KLDivergence: 0.6 - 0.8 succcess rate 231 bits / 0.344bpp
# last hidden layer (25088): MSE / CrossEntropy / KLDivergence: 0.45 - 0.7 succcess rate 231 bits / 0.344bpp
# last hidden layer+sigmoid (25088): MSE / CrossEntropy / KLDivergence:0.87; 0.91; 0.42 succcess rate 231 bits / 0.344bpp
# last hidden layer+sigmoid (25088): BCE: 0.89 succcess rate 231 bits / 0.344bpp

#tested: imagenet like cover (3x224x224); random resnet18 500 steps
# last layer (1000): MSE / CrossEntropy / KLDivergence: 0.6 - 0.8 succcess rate 231 bits / 0.344bpp
# last hidden layer (25088) with pretrained resnet18: MSE / CrossEntropy / KLDivergence: 0.45 - 0.7 succcess rate 231 bits / 0.344bpp
# last hidden layer +sigmoid (25088) with pretrained resnet18: MSE / CrossEntropy / KLDivergence: 0.65;0.65;0.65 succcess rate 231 bits / 0.344bpp



#tested: imagenet like cover (3x224x224); pretrained resnet18 with imagenet dataset 1000 steps; 10 random init; 0.05 step
# last layer (1000): MSE / CrossEntropy : 0.824;0.586 succcess rate 231 bits / 0.344bpp
# last hidden layer+sigmoid (25088): MSE / CrossEntropy /BCE: 0.97; 87.44;0.97 succcess rate 231 bits / 0.344bpp


#tested: imagenet like cover (3x224x224); random resnet18 with imagenet dataset 1000 steps; 10 random init; 0.05 step
# last layer (1000): MSE / CrossEntropy/BCE : 0.68;0.623; succcess rate 231 bits / 0.344bpp
# last hidden layer+sigmoid (25088): MSE / CrossEntropy /BCE:0.68; 0.67;0.67  succcess rate 231 bits / 0.344bpp


#tested: imagenet like cover (3x224x224); pretrained resnet18 with imagenet dataset 1000 steps; autopgd 10 random init;
# last layer (1000): MSE / CrossEntropy : 0.601;0.603 succcess rate 231 bits / 0.344bpp
# last hidden layer+sigmoid (25088): MSE / CrossEntropy /BCE: 0.993;0.69;0.994  succcess rate 231 bits / 0.344bpp
