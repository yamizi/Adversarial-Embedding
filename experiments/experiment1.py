"""
Generating Steganography images using Adversarial attacks 
"""



import sys
sys.path.append("./")
from experiments import logger, RANDOM_SEED

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.image import save_img, load_img, img_to_array
            
import numpy as np
import random, json, time, os
from utils.adversarial_models import load_model
from metrics.attacks import craft_attack

from PIL import Image
from stegano import lsb

experiment_time = int(time.time())
strs = "0123456789abcdefghijklmnopqrstuvwxyz"
default_path = "./experiments/results/experiment{}/pictures/{}_{}"
default_extension = "png"
palette = 256

def _encodeString(txt):
    base = len(strs)
    return str(int(txt, base))

def _decodeString(n):
    base = len(strs)
    
    if n < base:
      return strs[n]
    else:
      return _decodeString(n//base) + strs[n%base]

def _load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def _decode(dataset, model_type, epochs, experiment_id,attack_name, experiment_time, extension=None):
    if not extension:
        extension = default_extension
    pictures_path = default_path.format(experiment_id,attack_name, experiment_time)
    model, x_train, x_test, y_train, y_test = load_model(dataset=dataset, model_type=model_type, epochs=epochs)
    score = []
    for file in os.listdir(pictures_path):
        if file.endswith(".{}".format(extension)):
            path = "{}/{}".format(pictures_path,file)
            img = img_to_array(load_img(path))/palette
            img_class = np.argmax(model.predict(np.array([img]),verbose=0))
            index = file.index("_truth") -1
            real_class = int(file[index:index+1])
            steg_msg = lsb.reveal(path)
            logger.info("img {} decoded as {} stegano {}".format(file,img_class,steg_msg))
            
            score.append(real_class==img_class)

    decoding_score = np.mean(np.array(score))
    #logger.info("decoding score {}".format(decoding_score))
    return decoding_score

def _encode(msg,dataset, model_type, epochs, experiment_id,attack_name, keep_one=False, quality=100, attack_strength=2.0, extension=None):
    if not extension:
        extension = default_extension
    encoded_msg = _encodeString(msg)
    logger.info("Encode message {}=>{}".format(msg,encoded_msg))
    test_size = len(encoded_msg)
    model, x_train, x_test, y_train, y_test = load_model(dataset=dataset, model_type=model_type, epochs=epochs)
    num_classes= 10

    combined = list(zip(x_test, y_test))
    random.shuffle(combined)
    x_test[:], y_test[:] = zip(*combined)
    
    #keep only correctly predicted inputs
    batch_size = 64
    preds_test = np.argmax(model.predict(x_test,verbose=0), axis=1)
    inds_correct = np.where(preds_test == y_test.argmax(axis=1))[0]
    x, y = x_test[inds_correct], y_test[inds_correct]
    x, y = x[:test_size], y[:test_size]

    targets = np.array(to_categorical([int(i) for i in encoded_msg], num_classes), "int32")    
    #print(targets)
    
    if keep_one:
        x = np.repeat(np.array([x[0,:,:,:]]),y.shape[0], axis=0)
        y = model.predict(x)
    adv_x = craft_attack(model,x,attack_name,y=targets, epsilon=attack_strength)
    yadv = np.argmax(model.predict(adv_x), axis=1)
    
    pictures_path = default_path.format(experiment_id,attack_name, experiment_time)
    os.makedirs(pictures_path, exist_ok =True)
    os.makedirs("{}/ref".format(pictures_path), exist_ok =True)

    for i, adv in enumerate(adv_x):
        predicted = yadv[i]
        encoded = np.argmax(targets[i])
        truth = np.argmax(y[i])
        adv_path = "{}/{}_predicted{}_encoded{}_truth{}.{}".format(pictures_path,i,predicted,encoded,truth, extension)
        real_path = "{}/ref/{}.{}".format(pictures_path,i,extension)

        if extension=="png":
            q = int(10-quality/100)
            save_img(adv_path,adv, compress_level=q)
            save_img(real_path,x[i], compress_level=q)
        elif extension=="jpg":
            save_img(adv_path,adv, quality=quality)
            save_img(real_path,x[i], quality=quality)

    return experiment_time

def run(dataset="cifar10",model_type="basic", epochs = 25, ex_id="SP1"):

    attack_name = "targeted_pgd"
    logger.info("running {} {} {}".format(dataset,model_type, attack_name))
    

    if RANDOM_SEED>0:
        random.seed(RANDOM_SEED)

    quality=100
    extension = "png"
    nb_runs = 100

    experiment_id = "SP1/1"
    # msg1 = "Hello"
    # exp_time = _encode(msg1, dataset, model_type, epochs, experiment_id,attack_name,quality=quality, attack_strength=1.,extension = extension)
    # _decode( dataset, model_type, epochs, experiment_id,attack_name,exp_time,extension = extension)

    # experiment_id = "SP1/2"
    # exp_time = _encode(msg1, dataset, model_type, epochs, experiment_id,attack_name,quality=quality, attack_strength=1.,extension = extension,keep_one=True)
    # _decode( dataset, model_type, epochs, experiment_id,attack_name,exp_time,extension = extension)

    
    scores = []
    l = 100
    for i in range(nb_runs):
        logger.info(i)
        experiment_id = "{}/3_{}".format(ex_id,i)
        msg2 = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
        exp_time = _encode(msg2, dataset, model_type, epochs, experiment_id,attack_name,quality=quality, attack_strength=1.,extension = extension)
        score = _decode( dataset, model_type, epochs, experiment_id,attack_name,exp_time,extension = extension)
        scores.append(score)

    logger.info("{}:{}".format(experiment_id,np.array(scores).mean()))

    scores = []
    l = 100
    for i in range(nb_runs):
        
        experiment_id = "{}/4_{}".format(ex_id,i)
        msg2 = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
        exp_time = _encode(msg2, dataset, model_type, epochs, experiment_id,attack_name,quality=quality, attack_strength=1.,extension = extension,keep_one=True)
        score = _decode( dataset, model_type, epochs, experiment_id,attack_name,exp_time,extension = extension)
        scores.append(score)

    logger.info("{}:{}".format(experiment_id,np.array(scores).mean()))
    print()
    
if __name__ == "__main__":
    run(model_type="basic")