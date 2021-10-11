import sys, os, shutil
sys.path.append(".")


from comet_ml import Experiment
from utils import init_comet
from exploration.torch_layers import run as run_attack
import traceback
import numpy as np
import torch
from attacks import success_strategy
from exploration import get_model_dataset
import gc

import random

from exploration.yenet import trainYenet


def detect_xp(parameters,name):
    batch_loader, _ = get_model_dataset(parameters, device="cpu", return_loader=True)
    nb_batches = parameters.get("nb_batches", 5)
    validation_size = parameters.get("validation_size", 20)
    detector = parameters.get("detector","yenet")
    strategy = parameters.get("detector_strategy","blackbox")

    success_threshold = 0.65  # parameters["success_threshold"]

    noise_variance = 0.1

    experiment = None

    X = []
    Y = []
    ADV = []
    ADV_Y = []
    for i, batch in enumerate(batch_loader):
        batch_t = batch[0]
        if i>=nb_batches:
            #x_test = batch_t.float()
            break


        if strategy=="blackbox" or strategy=="graybox":
            max_eps = [8 / 255, 16 / 255, 0.1, 0.3, .5, 1]
            parameters["max_eps"] = random.choice(max_eps)
            if "cifar" in parameters["model"]:
                datasets = ["div2k","cifar10"]
                parameters["dataset"] = random.choice(datasets)
                models = ["cifar10_resnet20", "cifar10_resnet32"]
                parameters["model"] = random.choice(models)
            elif "imagenet" in parameters["model"]:
                datasets = ["div2k", "cifar10"]
                parameters["dataset"] = random.choice(datasets)
                models = ["imagenet_resnet18"]
                parameters["model"] = random.choice(models)

        print("batch", i)
        experiment, success1, x_test_adv, x_test, model, y_target , parameters = run_attack(parameters, name=name,
                                                                                        experiment=experiment,
                                                                                return_model=True, batch_t=batch_t,
                                                                                            batch_index=i)
        Y.append(y_target.cpu().detach())
        X.append(x_test.cpu().detach())
        ADV.append(x_test_adv.cpu().detach())


    decision_threshold = parameters.get("decision_threshold")
    absolute_capacity = parameters.get("message_size")


    y = torch.cat(Y).numpy() > decision_threshold
    clean_X = torch.cat(X)
    noised_X = clean_X + (noise_variance**0.5)*torch.rand(clean_X.size())
    adv = torch.cat(ADV)
    with torch.no_grad():
        predictions = model(adv.cuda())
        p = predictions.cpu().detach().numpy() > decision_threshold
        success = success_strategy(np.array(p, int), np.array(y, int), strategy=parameters.get("strategy"),
                                   data_size=absolute_capacity)


    is_adv = success > success_threshold
    if np.sum(is_adv)<len(is_adv)//4:
        is_adv = success >0.5

    index_adv = [i for i, x in enumerate(is_adv) if x]
    index_notadv = [i for i, x in enumerate(is_adv) if not x]
    adv_X = adv[is_adv]
    #partial_X = adv[~is_adv]
    nb_adv = len(adv_X)

    if strategy=="blackbox":
        sample = random.choices(index_notadv,k=nb_adv)
        X = [adv_X, clean_X[sample[:nb_adv//3]], noised_X[sample[nb_adv//3:2*nb_adv//3]], adv[sample[2*nb_adv//3:]]]
        DT = torch.cat(X,0).cuda()
        Y = torch.from_numpy(np.array([1]*nb_adv + [0]*nb_adv)).cuda()

    if strategy=="graybox" or strategy=="whitebox":
        X = [adv_X, clean_X[index_adv]]
        DT = torch.cat(X,0).cuda()
        Y = torch.from_numpy(np.array([1]*nb_adv + [0]*nb_adv)).cuda()


    shuffle = list(range(2*nb_adv))
    random.shuffle(shuffle)
    DT = DT[shuffle]
    Y = Y[shuffle]
    split_index = max(1,2*nb_adv*validation_size // 100)
    train_x , train_y =DT[split_index:], Y[split_index:]
    test_x , test_y =DT[:split_index], Y[:split_index]

    print(train_x.shape,train_y.shape, test_x.shape,test_y.shape)

    if detector=="yenet":
        detector = trainYenet(train_x,train_y, test_x=test_x, test_y=test_y, experiment=experiment)

    elif detector=="lcr":
        from attacks.adv_detector import trainLCR
        detector=trainLCR(train_x,train_y, model=model,test_x=test_x, test_y=test_y, experiment=experiment)


    del model
    gc.collect()
    torch.cuda.empty_cache()



if __name__ == '__main__':

    parameters = {"use_hidden": True, "pretrained": True, "criterion": "mse", "algorithm": "mifgsm", "max_eps": 0.3,
                  "norm": "Linf", "max_iter": 1000, "eps_step": 0.05, "num_random_init": 5, "batch_size": 10,
                  "success_threshold": 0.9, "bpp": 0.4, "lib": "torchattack", "model": "cifar10_resnet20",
                  "dataset": "cifar10", "strategy": "binary", "random_noise": "fading","binary_message":False, "transformation":None}

    parameters["validation_size"] = 20
    parameters["nb_batches"] = 1
    parameters["detector_epochs"] = 50
    #parameters["detector"] = "lcr"
    parameters["detector_strategy"] = "blackbox"

    name = "stegano_detection"
    #detect_xp(parameters, name=name)

    #exit()
    reload = True

    img_size = 256
    parameters["batch_size"] = 50
    parameters["detector_epochs"] = 50
    parameters["nb_batches"] = 4
    bpps = [0.4,1]
    threat_model = ["blackbox","graybox","whitebox"]


    parameters["model"] = "imagenet_resnet18"
    parameters["dataset"] = "div2k"

    #parameters["use_hidden"] = False
    #parameters["model"] = "imagenet_resnet18"
    #parameters["bpp"] = 0.03
    max_eps = [8/255,16/255,0.1,0.3,.5,1]
    max_eps = [8 / 255, 1]
    bpps = [1340 / img_size / img_size]
    name = "stegano_detection_imagenet"

    for threat in threat_model:
        parameters["detector_strategy"] = threat
        for eps in max_eps:
            parameters["max_eps"] = eps
            for bpp in bpps:
                parameters["bpp"] = bpp
                try:
                    detect_xp(parameters, name=name)
                except Exception as e:
                    print(e)
                    traceback.print_exc()



