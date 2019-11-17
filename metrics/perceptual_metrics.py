import sys, os
sys.path.append("./")
sys.path.append("./detecting-adversarial-samples/")
sys.path.append("./pyssim/")
sys.path.append("./lpips-tensorflow/")

import numpy as np
from metrics.attacks import craft_attack
from utils.adversarial_models import load_model

from PIL import Image
from matplotlib import cm
from ssim import compute_ssim
import lpips_tf

import numpy as np
from multiprocessing import Process, Pool

def lpips_distance(model, x, attack_name=None, adv_x = None, norm=2, distance_only=False, run_parallel=0):
    import tensorflow as tf
    if adv_x is None:
        adv_x = craft_attack(model, x,attack_name, norm)

    batch_size = 32
    image_shape = (batch_size, x.shape[1], x.shape[2], x.shape[3])
    image0_ph = tf.placeholder(tf.float32)
    image1_ph = tf.placeholder(tf.float32)

    distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex', model_dir="./metrics")

    def f(i):
        with tf.Session() as session:
            distance = session.run(distance_t, feed_dict={image0_ph: x[i], image1_ph: adv_x[i]})
            
            if distance_only:
                return distance
            else:
                return x[i], adv_x[i], distance

    if run_parallel>0:
        with Pool(run_parallel) as p:
            return p.map(f, range(len(x)))
    else:
        for i in range(len(x)):
            yield f(i)
    
def ssim_distance(model, x, attack_name=None, adv_x = None, norm=2, distance_only=False, run_parallel=0):

    if adv_x is None:
        adv_x = craft_attack(model, x,attack_name, norm)

    image = np.uint8(x*255)
    adv_image = np.uint8(adv_x*255)

    def f(i):
        img = Image.fromarray(image[i], 'RGB')
        adv_img = Image.fromarray(adv_image[i], 'RGB')
        distance = compute_ssim(img, adv_img)
        if distance_only:
            return distance
        else:
            return x[i], adv_x[i], distance

    if run_parallel>0:
        with Pool(run_parallel) as p:
            return p.map(f, range(len(x)))
    else:
        for i in range(len(image)):
            yield f(i)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
    import tensorflow as tf

    model, x_train, x_test, y_train, y_test = load_model(dataset="cifar10",model_type="basic",epochs=5)
    
    SSIM = np.array(list(ssim_distance(model, x_test[:128], "fgsm", distance_only=True)))
    print(SSIM.mean())

    SSIM = np.array(list(ssim_distance(model, x_test[:128], adv_x=x_test[:128], distance_only=True)))
    print(SSIM.mean())
    