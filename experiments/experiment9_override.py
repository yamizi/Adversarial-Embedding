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
from utils.adversarial_generator import AdversarialGenerator





def reset_memory():
    sess = get_session()
    clear_session()
    sess.close()

    # use the same config as you used to create the session
    config = tf.ConfigProto() #allow_soft_placement=True, log_device_placement=True)
    set_session(tf.Session(config=config))
    print("clean memory",gc.collect())


def run(dataset="cifar10",model_type="basic", epochs = 25, experiment_id="SP9c"):

    experiment_time = int(time.time())
    strs = "01"
    l = 511
    nb_messages = 1000

    folder = "./experiments/results/experiment{}".format(experiment_id)

    os.makedirs("{}".format(folder), exist_ok =True)

    nb_classes = [1]
    params = {'dataset': "cifar10",
          'shuffle': True, "model_epochs":"",
          'nb_elements':5000,'batch_size':192,"class_per_image":1}

    if RANDOM_SEED>0:
        random.seed(RANDOM_SEED)

    recovery_rates = []
    extension = "h5"
    models_path = "../products/run31c/cifar/ee50_te300_mr0.1_sr0.2_1565783786"
    skip_models = 0
    skip_models2 = 81
    experiment_id = "{}_{}".format(experiment_id,skip_models2)
    index = 0
    count = 2000
    checkpoint = 200

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


            train_msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
            Y_ref = np.array(list(AdversarialGenerator.encodeString(train_msg)),"int")

            Y_atk = ""
            while(len(Y_atk) !=len(Y_ref)):
                test_msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
                Y_atk = np.array(list(AdversarialGenerator.encodeString(test_msg)),"int")


            params["model_epochs"] = ""
            training_generator = AdversarialGenerator(train_msg, "train",model_type= model_src,**params)


            experiment_time = int(time.time())
            X = []

            for i, (x,y) in enumerate(training_generator.generate(plain=True)):
                if i == count:
                    break

                print("iter {}".format(i))
                X.append(x)

            X = np.array(X)

            with open("{}/{}.json".format(folder, exp_time), 'a') as f:
                f.write("[")

            index2 = 0
            for file in os.listdir(models_path):
                if max_models ==0 :#or file==src:
                    break
                if (file.startswith("e4") or file.startswith("e5") or file.startswith("e6") ) and file.endswith(".{}".format(extension)):
                    index2 = index2+1
                    if index2<=skip_models2 :
                        print("skipping {}".format(index2))
                        continue


                    model_type = "{}/{}".format(models_path,file)
                    max_models = max_models-1

                    params["model_epochs"] = 25
                    test_generator = AdversarialGenerator(test_msg, "train", model_type= model_type, **params)
                    test_generator.set = X.copy()
                    test_generator.shuffle = False
                    test_generator.adjust_batch_size()

                    X_override = []

                    for i, (x,y) in enumerate(test_generator.generate(plain=True)):

                        if i%len(Y_ref)==0:
                            if i>0:
                                X_override.append(X_)
                            X_ = []

                        if i == count:
                            break

                        #print("iter {}".format(i))
                        X_.append(x)


                    model, _, _, _, _ = load_model(dataset=params.get("dataset"), model_type=model_src, epochs=params.get("model_epochs"))

                    X_override = np.array(X_override)
                    Y_predicted = np.argmax(model.predict(X), axis=1)
                    Y_predicted_atk = [np.argmax(model.predict(_x), axis=1) for _x in X_override]
                    #y_override_msg = AdversarialGenerator.decodeString("".join([str(e) for e in Y_override]))


                    integrity = [sum(np.array(y_)==Y_atk)/len(Y_atk) for y_ in Y_predicted_atk]
                    availability = [sum(y_==Y_ref)/len(Y_ref) for y_ in Y_predicted_atk]

                    rate = {"model":file,"integrity":integrity, "availability":availability}
                    recovery_rates.append(rate)

                    with open("{}/{}.json".format(folder, exp_time), 'a') as f:
                        f.write("{},".format(json.dumps(rate)))

            with open("{}/{}.json".format(folder, exp_time), 'a') as f:
                    f.write("]")


    return



def analyze(models_path = "./experiments/results/experimentSP9_override"):

    import os, json
    values = {}

    model_name_size = 7

    for src in os.listdir(models_path):
      if src.endswith(".json"):
        index = src.rfind("_")
        model_name = src[index+1:-5]
        print("path {}".format(model_name))

        with open("{}/{}".format(models_path,src)) as f:

          fichier = f.read()
          try:
            vals = json.loads(fichier[:-1]+"]")
          except:
            vals = json.loads(fichier[:-2]+"]")
          integrity = np.array([e["integrity"] for e in vals if e["model"].find(model_name)==-1])
          val = {"min":integrity.min(), "max":integrity.max(), "mean":integrity.mean()}
          #values[model_name] = val
          print(val)

    print(len(values))


if __name__ == "__main__":
   #analyze()
   run(model_type="basic",experiment_id="SP9_override")
