"""
ART MLP Classifier
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append("./")
from experiments import logger, RANDOM_SEED, DATASET_CLASSES
import random

import time
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from utils.adversarial_generator import AdversarialGenerator

from keras.models import Sequential
from keras.layers import Dense

from PIL import Image

def _load_image( infilename ) :

    base_size = 32
    img = Image.open( infilename )
    img.load()
    img = img.resize((base_size,base_size))
    data = np.asarray( img, dtype="int32" )
    return data


def run(dataset="cifar10",model_type="basic", epochs = 50, exp_id="_gen_dataset"):

    # if RANDOM_SEED>0:
    #     random.seed(RANDOM_SEED)
    #     np.random.seed(RANDOM_SEED)

    imgs_path = "./utils/imagenet/train/images"
    strs = "01"
    l = 511
    nb_classes = [1]

    params = {'dataset': "cifar10",
          'shuffle': True,
          'nb_elements':5000,'batch_size':192,"class_per_image":1}

    train_msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
    training_generator = AdversarialGenerator(train_msg, "train",model_type= "basic",model_epochs=25, **params)

    set_shape = training_generator.set.shape
    print(set_shape)

    def truth_generator(max_val=0):
        i = 0
        for file in os.listdir(imgs_path):
            if max_val>0 and i>=max_val:
                break

            img_path = "{}/{}".format(imgs_path,file)
            image = _load_image(img_path)
            #print("building image {}: {}".format(i, image.shape))
            if len(image.shape)<3:
                continue
            i = i+1
            yield image

    truth_set = np.array(list(truth_generator(params['nb_elements'])))

    test_msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
    test_generator = AdversarialGenerator(test_msg, "test",model_type= "basic",model_epochs=25, **params)

    # detector model on dataset
    #detector.fit_generator(training_generator, use_multiprocessing=False, workers=5)
    count = 2*154
    checkpoint = 100
    experiment_time = int(time.time())
    default_path = "./experiments/results/sata/{}_{}/".format(params.get("class_per_image"),experiment_time)


    print("**** building test set")
    test_M, test_Y = [], []

    for i, (x,y, metric) in enumerate(test_generator.generate(detector_model_params={'dataset': "cifar10",'model_type':'basic'},truth_set=truth_set)):
        if i == count:
            np.save("./{}_testMY.npy".format(experiment_time),np.array([test_M,test_Y]))
            break

        print("===> test set iter {}".format(i))
        test_M.append(metric)
        test_Y.append(y)


    full_M, full_Y = [], []
    for i, (x,y, metric) in enumerate(training_generator.generate(detector_model_params={'dataset': "cifar10",'model_type':'basic'},truth_set=truth_set)):

        #if i%checkpoint==0:
        print("===> train set iter {}".format(i))
        if i == count:
            np.save("./{}_fullMY.npy".format(experiment_time),np.array([full_M,full_Y]))
            break
        full_Y.append(y)
        full_M.append(metric)




def run_detector(id="1581619212"):

    import tensorflow as tf
    from sklearn.metrics import roc_auc_score

    def auroc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

    test = (np.load("{}_testMY.npy".format(id), allow_pickle=True))
    train = (np.load("{}_trainMY.npy".format(id), allow_pickle=True))

    #print([len(e) for e in train[0]])
    #print(np.array(train[0]).shape,np.array(train[1]).shape,np.array(test[0]).shape,np.array(test[1]).shape)
    #return

    detector = Sequential()
    detector.add(Dense(100, input_dim=3, kernel_initializer='normal', activation='relu'))
    detector.add(Dense(2, kernel_initializer='normal', activation='sigmoid'))
    detector.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    x_train , y_train = np.array(list(train[0])), np.array(list(train[1]))
    x_test , y_test = np.array(list(test[0])), np.array(list(test[1]))

    predicts = detector.predict(x_test)
    predicts = predicts[:,1]
    print(y_test,predicts)


    detector.fit(x_train, y_train, batch_size = 64, epochs = 20,validation_split=0.2,shuffle=True)

    result = detector.evaluate(x_test,y_test,64)
    print("test",result)

if __name__ == "__main__":
    #run(model_type="basic")
    run_detector("1581623629")
