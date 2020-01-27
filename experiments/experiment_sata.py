"""
High Density embedding theory
52 bits message
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

import matplotlib.pyplot as plt

def _encodeString(txt):
    base = len(strs)
    return str(int(txt, base))


def run(dataset="cifar100",model_type="basic", epochs = 50, exp_id="SP_sata"):


    
    if RANDOM_SEED>0:
        # random.seed(RANDOM_SEED)
        # np.random.seed(RANDOM_SEED)
        pass

    
    l = 6643
    
    num_classes_tbl = (10,)
    #num_classes_tbl =[10000]
    nb_trials = 100
    nb_steps = 10

    nb_pictures = []
    nb_pixels = 32*32*3


    
    def plot(nb_pictures):
        for i,nb_picture in enumerate(nb_pictures):
            # plt.figure()
            # plt.title("Embedding density for N={}".format(num_classes_tbl[i]))
            # plt.xlabel('Nb classes embedded per image (k)')
            # plt.ylabel('Embedding Density (Bit Per Pixel)')

            density = l/ (nb_pixels * np.array(nb_picture))
            x = np.arange(len(nb_picture))+1
            x = x * (num_classes_tbl[i]/nb_steps)


            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel('Nb classes embedded per image (k)')
            ax1.set_ylabel('Embedding Density (Bits Per Pixel)', color=color)
            ax1.plot(x, density, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel('Number of images to embed the message', color=color)  # we already handled the x-label with ax1
            ax2.plot(x, nb_picture, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.title("Embedding density for N={}".format(num_classes_tbl[i]))
            #plt.plot(x,density)

        plt.show()

    for num_classes in num_classes_tbl:
        nb_pictures.append([])
        logger.info("total classes {}".format(num_classes))
        for nb_embedded_classes in range(1,nb_steps):
            num_embedded_classes = int(nb_embedded_classes*(num_classes/nb_steps))
            nb_imgs = []
            for j in range(nb_trials):
                msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
                encoded_msg = _encodeString(msg)
                logger.info("Encode message {}=>{}".format(msg,encoded_msg))

                groups = SATA.embed_message(None,None,encoded_msg, epsilon=None,class_density=num_embedded_classes/num_classes,num_classes=num_classes,groups_only=True)
                # logger.info("{}:{}".format(num_embedded_classes,len(groups)))
                nb_imgs.append(len(groups))
            nb_pictures[-1].append(np.array(nb_imgs).mean())


    
    #densities = l/ (nb_pixels * np.array(nb_pictures))

    plot(nb_pictures)

    folder = "./experiments/results/experiment{}".format("sata")
    os.makedirs(folder,exist_ok=True)
    with open("{}/{}.json".format(folder, "_".join(num_classes_tbl)), 'a') as f:
        f.write("{}".format(json.dumps(nb_pictures)))


if __name__ == "__main__":
    run()

