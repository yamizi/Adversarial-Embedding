"""
Detection of generated Steganography images using Adversarial attacks 
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
import random, json, time, os
from utils.adversarial_models import load_model
from metrics.attacks import craft_attack

from PIL import Image
from stegano import lsb

sys.path.append("./aletheia/")
from aletheialib import attacks, utils
alethia_model_paths = "./aletheia/models/"

from scipy import misc


experiment_time = int(time.time())
strs = "0123456789abcdefghijklmnopqrstuvwxyz"
default_path = "./experiments/results/experiment{}/pictures/{}_{}"
extension = "png"
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


def _detect_e4s_srm(files=None,file_path=None, model_file="e4s_srm_bossbase_lsbm0.10_gs.model", imgs=None):
    # 1 = clean, 0: stego

    
    from aletheialib import stegosim, feaext, models
    extractor="srm"

    clf=models.Ensemble4Stego()
    clf.load("{}{}".format(alethia_model_paths,model_file))

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

def _detect_spa(path):
    bitrate_R=attacks.spa(path, 0)
    bitrate_G=attacks.spa(path, 1)
    bitrate_B=attacks.spa(path, 2)

    threshold=0.05

    if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
        return 1
    else:
        return 0
        
def _decode(dataset, model_type, epochs, experiment_id,attack_name, experiment_time):
    pictures_path = default_path.format(experiment_id,attack_name, experiment_time)
    model, x_train, x_test, y_train, y_test = load_model(dataset=dataset, model_type=model_type, epochs=epochs)
    score = []
    lsb_score = []
    stat_score = []
    ensemble_score = []

    for file in os.listdir(pictures_path):
        if file.endswith(".{}".format(extension)):
            path = "{}/{}".format(pictures_path,file)
            img = img_to_array(load_img(path))/palette
            img_class = np.argmax(model.predict(np.array([img]),verbose=0))
            index = file.index("_truth") -1
            real_class = int(file[index:index+1])
            score.append(real_class==img_class)

            steg_msg = lsb.reveal(path)
            lsb_score.append(0 if lsb_score==None else 1)
            
            bitrate_R=attacks.spa(path, 0)
            bitrate_G=attacks.spa(path, 1)
            bitrate_B=attacks.spa(path, 2)

            threshold=0.05

            if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
                stat_score.append(0)
            else:
               stat_score.append(1) 
            #logger.info("img {} decoded as {} stegano {}".format(file,img_class,steg_msg))
            
    logger.info("decoding score {}".format(np.mean(np.array(score))))
    real_path = "{}/ref".format(pictures_path)
    for file in os.listdir(pictures_path):
        if file.endswith(".{}".format(extension)):
            path = "{}/{}".format(pictures_path,file)

            e4s_score = _detect_e4s_srm(file_path=path,model_file="e4s_srm_bossbase_lsbm0.10_gs.model")
            ensemble_score.append(e4s_score)


            steg_msg = lsb.reveal(path)
            lsb_score.append(1 if lsb_score==None else 0)
            
            spa_score = _detect_spa(path)
            stat_score.append(spa_score)

            
    lsb_score_mean = np.mean(np.array(lsb_score))
    stat_score_mean = np.mean(np.array(stat_score))
    
    logger.info("lsb detection score {}, stats detection score {}".format(lsb_score_mean, stat_score_mean))

        

def _encode(msg,dataset, model_type, epochs, experiment_id,attack_name, keep_one=False, quality=100, attack_strength=2.0):
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

def run(dataset="cifar10",model_type="basic", epochs = 25, experiment_id=5):

    attack_name = "targeted_pgd"
    logger.info("running {} {} {}".format(dataset,model_type, attack_name))
    

    if RANDOM_SEED>0:
        random.seed(RANDOM_SEED)

    l = 100
    msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])


    exp_time = _encode(msg, dataset, model_type, epochs, experiment_id,attack_name,quality=100, attack_strength=1.)
    _decode( dataset, model_type, epochs, experiment_id,attack_name,exp_time)
    # _decode( dataset, "resnet", epochs, experiment_id,attack_name,exp_time)
    # _decode( dataset, model_type, 5, experiment_id,attack_name,exp_time)
  
    # f1.write("{}".format(json.dumps(log)))
    # f1 = open("./experiments/results/experiment{}_{}.json".format(experiment_id,experiment_time), 'a')
    # f1.close()
    
if __name__ == "__main__":
    run(model_type="basic")