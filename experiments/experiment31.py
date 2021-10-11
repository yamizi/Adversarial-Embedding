"""
High Density attack recovery rate, detection (ATS) and similarity.
52 bits message in a 16x16 grayscale image 
"""



import sys
sys.path.append("./")
from experiments import logger, RANDOM_SEED, DATASET_CLASSES

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.image import save_img, load_img, img_to_array, array_to_img
            
import numpy as np
import random, json, time, os, math
from utils.adversarial_models import load_model
from metrics.attacks import craft_attack
from utils.sorted_attack import SATA

from PIL import Image
from metrics.perceptual_metrics import lpips_distance, ssim_distance

experiment_time = int(time.time())
strs = "01"
default_path = "./experiments/results/experiment{}/pictures/{}_{}"
default_extension = "png"
palette = 256


def _psnr_loss(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

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

def _encode(msg,dataset, model_type, epochs, experiment_id,attack_name, attack_strength=2.0):
    extension = default_extension
    encoded_msg = _encodeString(msg)
    logger.info("Encode message {}=>{}".format(msg,encoded_msg))
    test_size = len(encoded_msg)
    model, x_train, x_test, y_train, y_test = load_model(dataset=dataset, model_type=model_type, epochs=epochs, use_tensorboard=True)
    num_classes= DATASET_CLASSES[dataset]

    combined = list(zip(x_test, y_test))
    random.shuffle(combined)
    x_test[:], y_test[:] = zip(*combined)
    
    #keep only correctly predicted inputs
    batch_size = 64
    preds_test = np.argmax(model.predict(x_test,verbose=0), axis=1)
    inds_correct = np.where(preds_test == y_test.argmax(axis=1))[0]
    x, y = x_test[inds_correct], y_test[inds_correct]
    x, y = x[:test_size], y[:test_size]

    chunk_size = int(math.log(num_classes)/math.log(10))
    #groups = _split_msg(encoded_msg, chunk_size)

    targets = np.array(to_categorical([int(i) for i in encoded_msg], num_classes), "int32")    
    #print(targets)

    class_density = 0.03 # total class * density = total attacked classes
    epsilon = 5.0
    max_iter = 100
    SATA.power = 1.5
    nb_elements = 1000

    adv_x, rate_best = SATA.embed_message(model,x_test[:nb_elements],encoded_msg, epsilon=epsilon,class_density=class_density)
    
    #adv_x = craft_attack(model,x,attack_name,y=targets, epsilon=attack_strength)
    yadv = np.argmax(model.predict(adv_x), axis=1)
    
    pictures_path = default_path.format(experiment_id,attack_name, experiment_time)
    os.makedirs(pictures_path, exist_ok =True)
    os.makedirs("{}/ref".format(pictures_path), exist_ok =True)
    SSIM = []
    PSNR = []

    for i, _adv in enumerate(adv_x):
        predicted = yadv[i]
        encoded = np.argmax(targets[i])
        truth = np.argmax(y[i])
        adv_path = "{}/{}_predicted{}_encoded{}_truth{}.{}".format(pictures_path,i,predicted,encoded,truth, extension)
        real_path = "{}/ref/{}.{}".format(pictures_path,i,extension)
           
        adv = array_to_img(_adv)
        adv = adv.resize((16,16),Image.BILINEAR)
        #adv = adv.convert("L")
        adv.save(adv_path)

        adv_loaded = _load_image(adv_path)

        real_img = array_to_img(x[i])
        real_img = real_img.resize((16,16),Image.BILINEAR)
        #real_img = real_img.convert("L")
        #real = np.squeeze(img_to_array(real_img))
        real = img_to_array(real_img)

        ssim = 1-np.array(list(ssim_distance(None, np.array([real]), adv_x=np.array([adv_loaded]), distance_only=True)))
        SSIM.append(ssim)

        psnr = _psnr_loss(adv_loaded,real)
        PSNR.append(psnr)


    return np.array(SSIM).mean(), np.array(PSNR).mean()

def run(dataset="cifar100",model_type="basic", epochs = 50, exp_id="SP31"):

    attack_name = "targeted_pgd"
    
    if RANDOM_SEED>0:
        # random.seed(RANDOM_SEED)
        # np.random.seed(RANDOM_SEED)
        pass

    quality=100
    extension = "png"
    l = 52

    
    msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])

    experiment_id = "{}/1".format(exp_id)
    ssim, psnr = _encode(msg, dataset, model_type, epochs, experiment_id,attack_name)
    print(ssim,psnr)
    
    
if __name__ == "__main__":
    run(model_type="resnet", exp_id="test_{}".format(random.randint(0,10000)), epochs=100)
