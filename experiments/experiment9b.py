"""
Generating Steganography images using Adversarial attacks and comparing impact of data loss on recovery rate
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
from metrics.perceptual_metrics import lpips_distance, ssim_distance

from PIL import Image

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


def _decode(dataset, model_type, epochs, experiment_id,attack_name, experiment_time, extension=None, advs=None):
    if not extension:
        extension = default_extension
    pictures_path = default_path.format(experiment_id,attack_name, experiment_time)
    model, x_train, x_test, y_train, y_test = load_model(dataset=dataset, model_type=model_type, epochs=epochs)
    score = []

    if advs:
        for adv in advs:
            file = adv["file"]
            image = adv["img"]
            
            if len(image.size)<3:
                image = image.convert("RGB")
            if image.width<32:
                image = image.resize((32,32),Image.BILINEAR)
            img = img_to_array(image)/palette
            img_class = np.argmax(model.predict(np.array([img]),verbose=0))
            index = file.index("_truth") -1
            real_class = int(file[index:index+1])
            logger.info("img {} decoded as {}".format(file,img_class))
            
            score.append(real_class==img_class)
    else:
        for file in os.listdir(pictures_path):
            if file.endswith(".{}".format(extension)):
                path = "{}/{}".format(pictures_path,file)
                img = img_to_array(load_img(path))/palette
                img_class = np.argmax(model.predict(np.array([img]),verbose=0))
                index = file.index("_truth") -1
                real_class = int(file[index:index+1])
                logger.info("img {} decoded as {}".format(file,img_class))
                
                score.append(real_class==img_class)

    decoding_score = np.mean(np.array(score))
    logger.info("decoding score {}".format(decoding_score))
    return decoding_score

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

        if transformation=="color_depth":
            adv = adv.convert("P", palette=Image.ADAPTIVE, colors=8)

        elif transformation=="downscale":
            adv = adv.resize((16,16),Image.BILINEAR)

        elif transformation=="compress50":
            adv = _compress_img(adv,50)    

        elif transformation=="compress90":
            adv = _compress_img(adv,90)    

        yield  {"time":experiment_time,"file":adv_path,"img":adv}

        # adv.save(adv_path)

    
    return experiment_time

def run(dataset="cifar10",model_type="basic", epochs = 25, experiment_id="SP9b"):

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
    skip = 70
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

            x, y, adv_x, model, targets = _encode_adv(msg, dataset, model_type, epochs, experiment_id,attack_name,attack_strength=atk_strength)
            pre_computed = (x, y, adv_x, model, targets)
            SSIM = 1-np.array(list(ssim_distance(model, x, adv_x=adv_x, distance_only=True)))
            LPIPS = np.array(list(lpips_distance(model, x, adv_x=adv_x, distance_only=True)))
            PSNR = np.array([_psnr_loss(x[i],adv_x[i]) for i in range(len(x))])

        
            experiment_id = "SP9b/1"
            advs = _encode(msg, dataset, model_type, epochs, experiment_id,attack_name,attack_strength=atk_strength,extension = extension,transformation="compress90", pre_computed=pre_computed)
            compress90_recovery = _decode( dataset, model_type, epochs, experiment_id,attack_name,exp_time,extension = extension, advs=advs)

            experiment_id = "SP9b/2"
            advs = _encode(msg, dataset, model_type, epochs, experiment_id,attack_name,attack_strength=atk_strength,extension = extension,transformation="compress50", pre_computed=pre_computed)
            compress50_recovery = _decode( dataset, model_type, epochs, experiment_id,attack_name,exp_time,extension = extension, advs=advs)

            experiment_id = "SP9b/3"
            advs = _encode(msg, dataset, model_type, epochs, experiment_id,attack_name,attack_strength=atk_strength,extension = extension,transformation="downscale", pre_computed=pre_computed)
            downscale_recovery = _decode( dataset, model_type, epochs, experiment_id,attack_name,exp_time,extension = extension, advs=advs)

            experiment_id = "SP9b/4"
            advs = _encode(msg, dataset, model_type, epochs, experiment_id,attack_name,attack_strength=atk_strength,extension = "jpg",transformation="color_depth", pre_computed=pre_computed)
            color_depth_recovery = _decode( dataset, model_type, epochs, experiment_id,attack_name,exp_time,extension = "jpg", advs=advs)


            rate = {"model":file,"color_depth_recovery":color_depth_recovery,"downscale_recovery":downscale_recovery,"compress50_recovery":compress50_recovery,"compress90_recovery":compress90_recovery}
            rate = {**rate, "ssim":list(SSIM), "lpips":list(LPIPS),"psnr":list(PSNR)}
            recovery_rates.append(rate)

            with open("{}/{}.json".format(folder, experiment_time), 'a') as f:
                f.write("{},".format(json.dumps(rate)))

    logger.info(recovery_rates)
    with open("{}/{}.json".format(folder, experiment_time), 'a') as f:
            f.write("]")
    
    
    return 


    

    
if __name__ == "__main__":
    run(model_type="basic")