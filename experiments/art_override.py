"""
Adversarial embedding availability check (override attack)

availability 0.1255
"""



import sys
sys.path.append("./")
from experiments import logger, RANDOM_SEED, DATASET_CLASSES
import random
import os
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from utils.adversarial_generator import AdversarialGenerator
from utils.adversarial_models import load_model

from keras.models import Sequential
from keras.layers import Dense


def run(dataset="cifar10",model_type="basic", epochs = 50, exp_id="_gen_dataset"):

    # if RANDOM_SEED>0:
    #     random.seed(RANDOM_SEED)
    #     np.random.seed(RANDOM_SEED)

    strs = "01"
    l = 511
    nb_messages = 1000
    nb_classes = [1]
    params = {'dataset': "cifar10",
          'shuffle': True, "model_epochs":50,
          'nb_elements':5000,'batch_size':192,"class_per_image":1}

    train_msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
    Y_ref = np.array(list(AdversarialGenerator.encodeString(train_msg)),"int")

    Y_atk = ""
    while(len(Y_atk) !=len(Y_ref)):
        test_msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
        Y_atk = np.array(list(AdversarialGenerator.encodeString(test_msg)),"int")

    training_generator = AdversarialGenerator(train_msg, "train",model_type= "basic",**params)

    count = 2000
    checkpoint = 200
    experiment_time = int(time.time())
    X = []
    
    for i, (x,y) in enumerate(training_generator.generate(plain=True)):
        if i == count:
            break

        print("iter {}".format(i))
        X.append(x)

    X = np.array(X)
    
    params["model_epochs"] = 25
    test_generator = AdversarialGenerator(test_msg, "train", model_type= "resnet", **params) 
    test_generator.set = np.array(X)
    test_generator.shuffle = False
    test_generator.adjust_batch_size()


    X_override = []

    for i, (x,y) in enumerate(test_generator.generate(plain=True)):
        
        if i%len(Y_ref)==0:
            if i>0:
                X_override.append(X_)
            X_ = []

        if i == count:
            break

        print("iter {}".format(i))
        X_.append(x)
            
    model, _, _, _, _ = load_model(dataset=params.get("dataset"), model_type="basic", epochs=params.get("model_epochs"))

    X_override = np.array(X_override)
    print(X.shape, X_override.shape)
    

    Y_predicted = np.argmax(model.predict(X), axis=1)
    #y_msg = AdversarialGenerator.decodeString("".join([str(e) for e in Y]))

    Y_predicted_atk = [np.argmax(model.predict(_x), axis=1) for _x in X_override]
    #y_override_msg = AdversarialGenerator.decodeString("".join([str(e) for e in Y_override]))

    print("messages",len(Y_predicted_atk), Y_atk.shape, Y_ref.shape)

    default_path = "./experiments/results/override"
    os.makedirs(default_path, exist_ok =True)
    np.save("{}/Y_predicted_atk.npy".format(default_path),Y_predicted_atk)
    np.save("{}/Y_ref.npy".format(default_path),Y_ref)
    np.save("{}/Y_atk.npy".format(default_path),Y_atk)

    # closeness = sum(Y_predicted==Y_predicted_atk) / len(Y_predicted_atk)
    # print("closeness {}".format(closeness),Y_predicted.shape,Y_predicted_atk.shape)

    

if __name__ == "__main__":
    #run(model_type="basic")

    
    default_path = "./experiments/results/override"
    Y_predicted_atk = np.load("{}/Y_predicted_atk.npy".format(default_path))
    Y_ref = np.load("{}/Y_ref.npy".format(default_path))
    Y_atk = np.load("{}/Y_atk.npy".format(default_path))
    
    integrity = np.array([sum(np.array(y_)==Y_atk)/len(Y_atk) for y_ in Y_predicted_atk])
    availability = np.array([sum(y_==Y_ref)/len(Y_ref) for y_ in Y_predicted_atk])
    print("integrity",integrity.mean(),availability.mean())
