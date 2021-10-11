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
import time

from PIL import Image
from metrics.perceptual_metrics import lpips_distance, ssim_distance

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

def _encode(msg,dataset, model_type, epochs, experiment_id,attack_name, attack_strength=5.0, nb_classes=2,experiment_time=""):
    print(dataset,model_type,epochs)
    extension = default_extension
    encoded_msg = _encodeString(msg)
    logger.info("Encode message {}=>{}".format(msg,encoded_msg))
    test_size = len(encoded_msg)
    model, x_train, x_test, y_train, y_test = load_model(dataset=dataset, model_type=model_type, epochs=epochs)
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

    epsilon = 5.0
    max_iter = 100
    SATA.power = 1.5
    nb_elements = 1000

    sub_x = x_test[:nb_elements]
    sub_y = y_test[:nb_elements]

    begin = time.time()
    adv_x, ref_x, rate_best = SATA.embed_message(model,sub_x,encoded_msg, epsilon=epsilon,nb_classes_per_img=nb_classes)
    ref_y = np.argmax(model.predict(ref_x,verbose=0), axis=1)
    end= time.time()
    
    #adv_x = craft_attack(model,x,attack_name,y=targets, epsilon=attack_strength)
    adv_y = np.argmax(model.predict(adv_x), axis=1)
    nb_required_imgs = adv_y.shape[0]
    stats = [end, begin,msg,nb_required_imgs,nb_classes, epsilon, max_iter,SATA.power,nb_elements]
    print("nb images required: {}".format(nb_required_imgs))
    
    pictures_path = default_path.format(experiment_id,attack_name, experiment_time)
    os.makedirs(pictures_path, exist_ok =True)

    np.save("{}/ref_x.npy".format(pictures_path),ref_x)
    np.save("{}/ref_y.npy".format(pictures_path),ref_y)
    np.save("{}/adv_x.npy".format(pictures_path),adv_x)
    np.save("{}/adv_y.npy".format(pictures_path),adv_y)
    np.save("{}/stats.npy".format(pictures_path),np.array(stats))
    
def run(dataset="cifar10",model_type="basic", epochs = 25, exp_id="_gen_dataset"):

    attack_name = "targeted_pgd"
    
    if RANDOM_SEED>0:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        pass

    quality=100
    extension = "png"
    l = 52
    nb_messages = 1000

    nb_classes = [2,3]
    #nb_classes = [4,5]
    for nb_cls in nb_classes:
        experiment_id = "{}/{}".format(exp_id,nb_cls)
        for i in range(nb_messages):
            logger.info("## class {} iter {}".format(nb_cls, i))
            experiment_time = int(time.time())
            msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
            _encode(msg, dataset, model_type, epochs, experiment_id,attack_name,nb_classes=nb_cls,experiment_time=experiment_time)
    
    
if __name__ == "__main__":
    run(model_type="basic")
