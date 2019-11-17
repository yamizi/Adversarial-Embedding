"""
Detecting Steganography images using Adversarial attacks and comparing impact of data loss on recovery rate
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
from utils.adversarial_models import load_model
from metrics.attacks import craft_attack

sys.path.append("./aletheia/")
from aletheialib import attacks, utils
alethia_model_paths = "./aletheia/models/"


from PIL import Image
import h5py

experiment_time = int(time.time())
strs = "0123456789abcdefghijklmnopqrstuvwxyz"
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

def _compress_img(img, rate=75, format='jpeg', palette=256):
    from io import BytesIO
    
    
    byteImgIO  = BytesIO()
    img.save(byteImgIO , format=format,quality=rate)
    byteImgIO.seek(0)

    dataBytesIO = BytesIO(byteImgIO.read())
    compressed_img = Image.open(dataBytesIO)

    return compressed_img


def _detect_spa(files=None,file_path=None, imgs=None):

    if not files and file_path:
        files = [file_path]

    
    threshold=0.05
    stat_score = []
    
    for i, f in enumerate(files):
        img = imgs[i] if imgs is not None else None
        bitrate_R=attacks.spa(f, 0, img)
        bitrate_G=attacks.spa(f, 1, img)
        bitrate_B=attacks.spa(f, 2, img)

        if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
            stat_score.append(1)
        else:
            stat_score.append(0) 

    np.array(stat_score)

def _detect_e4s_srm(files=None,file_path=None, model_file="e4s_srm_bossbase_lsbm0.10_gs.model", imgs=None):
    # 1 = clean, 0: stego

    
    from aletheialib import stegosim, feaext, models
    extractor="srm"

    clf=models.Ensemble4Stego()
    clf.load(model_file.format(alethia_model_paths))

    stat_score = []

    if not files and file_path:
        files = [file_path]

    for i, f in enumerate(files):
        
        img = array_to_img(imgs[i]) if imgs else None
        X = feaext.extractor_fn(extractor)(f,im=img)
        X = X.reshape((1, X.shape[0]))
        p = clf.predict(X)
        if p[0] == 0:
            stat_score.append(1)
        else:
            stat_score.append(0) 

    return np.array(stat_score)
            
def _encode_adv(msg,dataset, model_type, epochs, experiment_id,attack_name, attack_strength=2.0):

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
    
    adv_x = craft_attack(model,x,attack_name,y=targets, epsilon=attack_strength)

    return x, y, adv_x, model, targets

def run(dataset="cifar10",model_type="basic", epochs = 25, experiment_id="SP9_6"):

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
    max_models = 150
    skip = 0
    with open("{}/{}.json".format(folder, experiment_time), 'a') as f:
        f.write("[")
            
    for file in os.listdir(models_path):
        if max_models ==0:
            break
        if (file.startswith("e4") or file.startswith("e5") or file.startswith("e6") ) and file.endswith(".{}".format(extension)):
            
            if skip>0:
                skip = skip-1
                continue

            max_models = max_models-1
            model_type = "{}/{}".format(models_path,file)
            exp_time = 0
            atk_strength = 5.0

            x, y, adv_x, model, targets = _encode_adv(msg, dataset, "basic", epochs, experiment_id,attack_name,attack_strength=atk_strength)
            
            pic_folder = "{}/{}".format(folder,file )
            os.makedirs(pic_folder, exist_ok =True)
            files = []
            
            for i,adv in enumerate(adv_x):
               adv_path = "{}/{}.{}".format(pic_folder,i,extension)
               #save_img(adv_path,adv)
               #array_to_img(adv).save(adv_path)
               files.append(adv_path)
               
            rate = {"model":file}
            rate["stat_score_adv"] = _detect_spa(files,imgs=adv_x)
            rate["stat_score_real"] = _detect_spa(files,imgs=x)

            # rate["lsb_score_adv"] = _detect_e4s_srm(files, model_file="e4s_srm_bossbase_lsbm0.40_gs.model",imgs=adv_x)
            # rate["lsb_score_real"] = _detect_e4s_srm(files, model_file="e4s_srm_bossbase_lsbm0.40_gs.model",imgs=x)
            # rate["hill_score"] = _detect_e4s_srm(files, model_file="e4s_srm_bossbase_hill0.40_gs.model")
            # rate["wow_score"] = _detect_e4s_srm(files, model_file="e4s_srm_bossbase_wow0.40_gs.model")
            # rate["uniw_score"] = _detect_e4s_srm(files, model_file="e4s_srm_bossbase_uniw0.40_gs.model")
            
            recovery_rates.append(rate)

            with open("{}/{}.json".format(folder, experiment_time), 'a') as f:
                f.write("{},".format(json.dumps(rate)))

    logger.info(recovery_rates)
    with open("{}/{}.json".format(folder, experiment_time), 'a') as f:
            f.write("]")
    
    
    return 


    

    
if __name__ == "__main__":
    run(model_type="basic")