"""
ART MLP Classifier
"""



import sys
sys.path.append("./")
from experiments import logger, RANDOM_SEED, DATASET_CLASSES
import random
import os
import time
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from utils.adversarial_generator import AdversarialGenerator

from keras.models import Sequential
from keras.layers import Dense


def run(dataset="cifar10",model_type="basic", epochs = 50, exp_id="_gen_dataset"):

    attack_name = "targeted_pgd"
    
    # if RANDOM_SEED>0:
    #     random.seed(RANDOM_SEED)
    #     np.random.seed(RANDOM_SEED)

    strs = "01"
    l = 511
    nb_messages = 1000
    nb_classes = [1]
    #nb_classes = [4,5]

    detector = Sequential()
    detector.add(Dense(100, input_dim=3, kernel_initializer='normal', activation='relu'))
    detector.add(Dense(2, kernel_initializer='normal', activation='sigmoid'))   
    detector.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    params = {'dataset': "cifar10",
          'shuffle': True,
          'nb_elements':5000,'batch_size':192,"class_per_image":1}

    train_msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
    training_generator = AdversarialGenerator(train_msg, "train",model_type= "basic",**params)

    test_msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
    test_generator = AdversarialGenerator(test_msg, "test",model_type= "basic", **params) 

    # detector model on dataset
    #detector.fit_generator(training_generator, use_multiprocessing=False, workers=5)
    count = 2000
    checkpoint = 200
    experiment_time = int(time.time())
    default_path = "./experiments/results/sata/{}_{}/".format(params.get("class_per_image"),experiment_time)

    full_M, full_Y = [], []

    
    for i, (x,y, metric) in enumerate(training_generator.generate()):

        if i%checkpoint==0:
            if i>0:
                save_path = "{}/{}".format(default_path,i)
                os.makedirs(save_path, exist_ok =True)
                np.save("{}/x.npy".format(save_path),X)
                np.save("{}/y.npy".format(save_path),Y)
                np.save("{}/m.npy".format(save_path),M)
            X = []
            Y = []
            M = []
        
        if i == count:
            break
        print("iter {}".format(i))
        X.append(x)
        Y.append(y)
        full_Y.append(y)
        M.append(metric)
        full_M.append(metric)
        

    
    detector.fit(np.array(full_M), np.array(full_Y), batch_size = 64, epochs = 20,validation_split=0.2,shuffle=True)
    nb_test = 1000
    test_M, test_Y = [], []

    for k in range(nb_test):
        x,y,metric = test_generator.generate()
        test_M.append(metric)
        test_Y.append(y)

    result = detector.evaluate(test_M,test_Y,64)
    print("test",result)
    
if __name__ == "__main__":
    run(model_type="basic")