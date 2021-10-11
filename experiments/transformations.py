import sys
sys.path.append(".")


from comet_ml import Experiment
from utils import init_comet
from exploration.torch_layers import run as run_attack
import traceback
import numpy as np
import torch
from attacks import success_strategy
from exploration import compress_img, depth_img
import gc
from PIL import Image as pil_image
import matplotlib.pyplot as plt
from torchvision import transforms as torchtrf

def transform(images_tensor,transformation):
    transformation = transformation.split("#")

    transformed_adv = images_tensor.cpu().detach().numpy()
    images = [e.swapaxes(0, 1).swapaxes(1, 2)*255 for e in transformed_adv]
    images = [pil_image.fromarray(e.astype('uint8') , 'RGB') for e in images]
    images = [torchtrf.ToPILImage()(e) for e in images_tensor] 

    if transformation[0]=="rotate":
        transformed = [(e.rotate(int(transformation[1]))) for e in images]

    elif transformation[0]=="compress":
        transformed = [(compress_img(e,int(transformation[1]))) for e in images]

    elif transformation[0]=="depth":
        transformed = [(depth_img(e, int(transformation[1]))) for e in images]
    elif transformation[0] == "scale":
        target_size = (int(transformation[1]), int(transformation[1]))
        original_size = transformed_adv[0].shape[1:]
        transformed = [(e.resize(target_size).resize(original_size)) for e in images]

    elif transformation[0] == "crop":
        transformation_params = [int(i) for i in transformation[1].split(",")]
        transformed = [np.asarray(e.crop(transformation_params))for e in images]

    else:
        transformed = images
        
    transformed_t = [torchtrf.ToTensor()(e).unsqueeze(0).cuda() for e in transformed]
    x_test_adv = torch.cat(transformed_t,0)
    print(x_test_adv.shape)
    return x_test_adv, images


    transformed_images = [e.swapaxes(1, 2).swapaxes(0, 1)/255 for e in transformed]

    x_test_adv = torch.from_numpy(np.array(transformed_images)).float().cuda()

    return x_test_adv, images

def transform_xp(parameters,name):
    experiment, success1, x_test_adv, x_test, model, y_target , parameters = run_attack(parameters, name=name, return_model=True)

    #plt.imshow(x_test_adv[0].cpu().numpy().swapaxes(0, 1).swapaxes(1, 2))
    #plt.show()
    #plt.imshow(x_test[0].cpu().numpy().swapaxes(0, 1).swapaxes(1, 2))
    #plt.show()
    print(len(x_test_adv))
    print(x_test_adv.shape)
    x_test_adv_transformed, images = transform(x_test_adv, parameters["transformation"])

    with torch.no_grad():
        predictions_transform = model(x_test_adv_transformed)
        predictions = model(x_test_adv)

    decision_threshold = parameters["decision_threshold"]
    absolute_capacity = parameters["message_size"]
    p = predictions_transform.cpu().detach().numpy() > decision_threshold
    y = predictions.cpu().detach().numpy() > decision_threshold
    success = success_strategy(p,y,strategy=parameters.get("strategy"),data_size=absolute_capacity)

    print("success strategy adv",success)

    for (i,s) in enumerate(success):
        experiment.log_metric("accuracy_transform", success[i], step=i)
        experiment.log_metric("rs-bpp_transformed", (2 * success[i] - 1) * parameters.get("density"), step=i)
        experiment.log_image(images[i], name='Val/image transformed ', image_channels='first', step=i)

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    nb_classes_to_attack = 25
    parameters = {"use_hidden": True, "pretrained": True, "criterion": "mse", "algorithm": "mifgsm", "max_eps": 0.1,
                  "norm": "Linf", "max_iter": 1000, "eps_step": 0.05, "num_random_init": 5, "batch_size": 10,
                  "success_threshold": 0.9, "bpp": 0.4, "lib": "torchattack", "model": "cifar10_resnet20",
                  "dataset": "cifar10", "strategy": "binary", "random_noise": "fading","binary_message":False, "transformation":"rotate#10"}

    name = "stegano_transformations_dgx"
    #transform_xp(parameters, name=name)

    #exit()
    reload = True

    parameters["batch_size"] = 100
    transforms = ["rotate#10","crop#2,2,30,30","scale#16","depth#8", "scale#64",  "compress#50", "compress#75", "compress#90"]

    img_size = 32
    img_size = 256
    bpps = [408/img_size/img_size,1000/img_size/img_size]
    bpps = [1340 / img_size / img_size]
    #parameters["use_hidden"] = False
    parameters["model"] = "imagenet_resnet18"
    parameters["dataset"] = "div2k"
    #parameters["bpp"] = 0.03
    max_eps = [8/255,16/255,0.1,0.3,.5,1]
    max_eps = [8 / 255, 1]
    name = "stegano_transform_imagenet"

    for transform_val in transforms:
        parameters["transformation"] = transform_val
        for eps in max_eps:
            parameters["max_eps"] = eps
            for bpp in bpps:
                parameters["bpp"] = bpp
                try:
                    transform_xp(parameters, name)
                except Exception as e:
                    print(e)
                    traceback.print_exc()



