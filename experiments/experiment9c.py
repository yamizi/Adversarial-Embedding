"""
Decoding Steganography images using Adversarial attacks and comparing impact of data loss on recovery rate
using different models
"""



import sys
sys.path.append("./")
from experiments import logger, RANDOM_SEED

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.image import save_img, load_img, img_to_array, array_to_img
            
import numpy as np
import random, json, time, os, math
from utils.adversarial_models import load_model, load_dataset
from metrics.attacks import craft_attack
import gc
from keras.backend.tensorflow_backend import set_session, clear_session, get_session

from PIL import Image

experiment_time = int(time.time())
strs = "0123456789abcdefghijklmnopqrstuvwxyz"
default_path = "./experiments/results/experiment{}/pictures/{}_{}"
default_extension = "png"
palette = 256


def reset_memory():
    sess = get_session()
    clear_session()
    sess.close()

    # use the same config as you used to create the session
    config = tf.ConfigProto() #allow_soft_placement=True, log_device_placement=True)
    set_session(tf.Session(config=config))
    print("clean memory",gc.collect())
        


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

def _compress_img(img, rate=75, format='jpeg', palette=256):
    from io import BytesIO
    
    
    byteImgIO  = BytesIO()
    img.save(byteImgIO , format=format,quality=rate)
    byteImgIO.seek(0)

    dataBytesIO = BytesIO(byteImgIO.read())
    compressed_img = Image.open(dataBytesIO)

    return compressed_img


def _decode(dataset, model_type, epochs, experiment_id,attack_name, experiment_time, extension=None, advs=None):
    if not extension:
        extension = default_extension
    model, x_train, x_test, y_train, y_test = load_model(dataset=dataset, model_type=model_type, epochs=epochs)
    score = []

    for adv in advs:
        real_class = adv["class"]
        image = adv["img"]
        
        if len(image.size)<3:
            image = image.convert("RGB")
        if image.width<32:
            image = image.resize((32,32),Image.BILINEAR)
        img = img_to_array(image)/palette
        img_class = np.argmax(model.predict(np.array([img]),verbose=0))
        logger.info("img {} decoded as {}".format(real_class,img_class))
        
        score.append(real_class==img_class)

    decoding_score = np.mean(np.array(score))
    logger.info("decoding score {}".format(decoding_score))
    return decoding_score

def _encode_adv(msg,dataset, model_type, epochs, experiment_id,attack_name, attack_strength=2.0, model=None):

    encoded_msg = _encodeString(msg)
    logger.info("Encode message {}=>{}".format(msg,encoded_msg))
    test_size = len(encoded_msg)

    if model is None:
        model, x_train, x_test, y_train, y_test = load_model(dataset=dataset, model_type=model_type, epochs=epochs)
    else:
        num_classes, x_train, y_train, x_test, y_test = load_dataset(dataset=dataset)
        
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
    
    adv_x = craft_attack(model,x,attack_name,y=targets, epsilon=attack_strength)

    return x, y, adv_x, model, targets

def _encode(msg,dataset, model_type, epochs, experiment_id, attack_name, attack_strength=2.0, extension=None, transformation=None, pre_computed=None):
    if not extension:
        extension = default_extension

    if pre_computed is None:
        x, y, adv_x, model, targets = _encode_adv(msg,dataset, model_type, epochs, experiment_id,attack_name, attack_strength)
    else:
        x,y, adv_x, model, targets = pre_computed[0], pre_computed[1], pre_computed[2], pre_computed[3], pre_computed[4]

    yadv = np.argmax(model.predict(adv_x), axis=1)
    
    pictures_path = default_path.format(experiment_id,attack_name, experiment_time)
    os.makedirs(pictures_path, exist_ok =True)
    os.makedirs("{}/ref".format(pictures_path), exist_ok =True)

    advs = []
    for i, _adv in enumerate(adv_x):
        predicted = yadv[i]
        encoded = np.argmax(targets[i])
        truth = np.argmax(y[i])
        adv_path = "{}/{}_predicted{}_encoded{}_truth{}.{}".format(pictures_path,i,predicted,encoded,truth, extension)
        real_path = "{}/ref/{}.{}".format(pictures_path,i,extension)
        
        adv = array_to_img(_adv)
  

        yield  {"time":experiment_time,"file":adv_path,"img":adv}

    return experiment_time

def _transmit(adv_x,y):
    for index, _adv in enumerate(adv_x):
        img_adv = array_to_img(_adv)
        yield {"class":np.argmax(y[index]),"img":img_adv}
        
def run(dataset="cifar10",model_type="basic", epochs = 25, experiment_id="SP9c"):

    attack_name = "targeted_pgd"
    logger.info("running {} {} {}".format(dataset,model_type, attack_name))
    folder = "./experiments/results/experiment{}".format(experiment_id)

    os.makedirs("{}".format(folder), exist_ok =True)

    if RANDOM_SEED>0:
        random.seed(RANDOM_SEED)

    quality=100
    extension = "png"
    l = 100    
    msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])

    recovery_rates = []
    extension = "h5"
    models_path = "../products/run31c/cifar/ee50_te300_mr0.1_sr0.2_1565783786"
    skip_models = 90
    index = 0
            
    for src in os.listdir(models_path): 
        if (src.startswith("e4") or src.startswith("e5") or src.startswith("e6") ) and src.endswith(".{}".format(extension)):
            
            reset_memory()
            index = index + 1 
            if index<=skip_models :
                continue

            max_models = 100
            model_src = "{}/{}".format(models_path,src)
            exp_time = "{}_{}".format(experiment_time,src[:10])
            atk_strength = 2.0

            x, y, adv_x, model, targets = _encode_adv(msg, dataset, model_src, epochs, experiment_id,attack_name,attack_strength=atk_strength)
            pre_computed = (x, y, adv_x, model, targets)
            
            advs = list(_transmit(adv_x,targets))

            with open("{}/{}.json".format(folder, exp_time), 'a') as f:
                f.write("[")

            for file in os.listdir(models_path):
                if max_models ==0 :#or file==src:
                    break
                if (file.startswith("e4") or file.startswith("e5") or file.startswith("e6") ) and file.endswith(".{}".format(extension)):
                    
                    model_type = "{}/{}".format(models_path,file)
                    max_models = max_models-1
                    
                    rate_recovery = _decode( dataset, model_type, epochs, experiment_id,attack_name,exp_time,extension = extension, advs=advs)
                    
                    rate = {"model":file,"decoding_recovery":rate_recovery}
                    recovery_rates.append(rate)

                    with open("{}/{}.json".format(folder, exp_time), 'a') as f:
                        f.write("{},".format(json.dumps(rate)))

            logger.info(recovery_rates)
            with open("{}/{}.json".format(folder, exp_time), 'a') as f:
                    f.write("]")
    return 


    
if __name__ == "__main__":
    run(model_type="basic",experiment_id="SP9c_grid")
