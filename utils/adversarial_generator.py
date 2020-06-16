import numpy as np
import keras

from keras.utils import to_categorical

import random, json, time, os, math
from utils.adversarial_models import load_model, load_dataset
from metrics.attacks import craft_attack
from utils.sorted_attack import SATA
import time

from PIL import Image
from metrics.perceptual_metrics import lpips_distance, ssim_distance
from metrics.uncertainty_metrics import get_uncertain_predictions
from detect.util import get_noisy_samples

class AdversarialGenerator(keras.utils.Sequence):

    strs = "01"

    

    def encodeString(txt):
        base = len(AdversarialGenerator.strs)
        return str(int(txt, base))

    def decodeString(n):
        n= int(n)
        base = len(AdversarialGenerator.strs)
        
        if n < base:
            return AdversarialGenerator.strs[n]
        else:
            return AdversarialGenerator.decodeString(n//base) + AdversarialGenerator.strs[n%base]
    

    def _encode(self, sub_x):
       
        epsilon = 2.0
        max_iter = 1#0
        SATA.power = 2
        SATA.num_cover_init = 2
        begin = time.time()

        model, _, _, _, _ = load_model(**self.model_params)

        adv_x, ref_x, rate_best = SATA.embed_message(model,sub_x,self.secret_message, epsilon=epsilon,nb_classes_per_img=self.class_per_image)
        end= time.time()
        
        return adv_x, ref_x, rate_best, model
        
    'Generates data for Keras'
    def __init__(self, secret_msg,split,dataset, model_type, class_per_image=1, model_epochs=50, batch_size=128, nb_elements = 1000, shuffle=True):
        'Initialization'

        num_classes, _, _, _,_ = load_dataset(dataset=dataset)
        model,x_train, x_test, y_train, y_test  = load_model(dataset=dataset, model_type=model_type, epochs=model_epochs)

        if split=="train":
            x = x_train
            y = y_train
        else:
            x = x_test
            y = y_test

        
        
        #keep only correctly predicted inputs
        preds_test = np.argmax(model.predict(x,verbose=0), axis=1)
        #print("-----",x.shape,y.shape,preds_test.shape)
        inds_correct = preds_test == y.argmax(axis=1)
        x,y = x[inds_correct], y[inds_correct]
        
        sub_x = x[:nb_elements]
        sub_y = y[:nb_elements]

        self.list_IDs = list(range(nb_elements))
    
        self.secret_message = AdversarialGenerator.encodeString(secret_msg)
        self.model_params = dict(dataset=dataset, model_type=model_type, epochs=model_epochs)
        self.set = sub_x
        self.labels = sub_y
        self.dim = (x_train.shape[0],x_train.shape[1])
        self.batch_size = batch_size
        self.n_channels = x_train.shape[2]
        self.n_classes = num_classes
        self.class_per_image = class_per_image
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #print("X",index,indexes)
        

        # Generate data
        X, y = self._data_generation(indexes)# list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.set))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        adv_x, ref_x, rate_best, model = self._encode(self.set[indexes,:])
        ref_y = model.predict(ref_x)

        nb_elements = adv_x.shape[0]
        #self.list_IDs = list(range(nb_elements))
        
        d = "cifar" if self.model_params.get("dataset")=="cifar10" else self.model_params.get("dataset")
        atk = "bim-b"
        noisy_x =  get_noisy_samples(ref_x, adv_x, d, atk)
        
        X = np.concatenate(([adv_x],[noisy_x]),axis=1)
        y = np.concatenate(([np.ones(nb_elements)] ,[np.zeros(nb_elements)]),axis=1)
        y= keras.utils.to_categorical(y, num_classes=self.n_classes)
        X = np.squeeze(X)
        y = np.squeeze(y)
        print("****X",X.shape,y.shape)
        
        lcr, variance, kde = get_uncertain_predictions(model, np.array([adv_x,noisy_x]), ref_x,ref_y)

        print("****LCR",lcr.shape,variance.shape,kde.shape)
        return X, y


        #print("*******",len(self.list_IDs), X.shape,y.shape)#,lcr.shape,variance.shape,kde.shape)

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def adjust_batch_size(self):
        model, _, _, _, _ = load_model(**self.model_params)
        groups = SATA.embed_message(model,None,self.secret_message, nb_classes_per_img=self.class_per_image,groups_only=True)
        self.batch_size = len(groups)

    def generate(self,count=10, plain=False, detector_model=None, truth_set=None,detector_model_params=None):
        while True:
            for index in range(count):
                self.on_epoch_end()
                #print("--> index",index)
                # Generate indexes of the batch
                indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
                #print("*** indexes", indexes)

                #print("indexes",indexes.shape)


                adv_x, ref_x, rate_best, model = self._encode(self.set[indexes,:])
                ref_y = model.predict(ref_x)

                if plain:
                    X = adv_x
                    y = ref_y
                    for i,x in enumerate(X):
                        #print("plain")
                        yield x, y[i]
                else:
                    if detector_model is None:
                        detector_model = model
                    else:
                        print("adversarial model",detector_model_params)
                        model_params = self.model_params
                        model_params["dataset"] = detector_model_params["dataset"]
                        model_params["model_type"] = detector_model_params["model_type"]

                        detector_model, _, _, _, _ = load_model(**model_params)

                    nb_elements = adv_x.shape[0]
                    #self.list_IDs = list(range(nb_elements))

                    if truth_set is None:
                        ground_truth = self.set[indexes,:]
                    else:
                        ground_truth = truth_set[indexes,:]

                    ground_truth = ground_truth[0:len(adv_x),:]
                    X = np.concatenate(([adv_x],[ground_truth]),axis=1)
                    y = np.concatenate(([np.zeros(nb_elements)] ,[np.ones(nb_elements)]),axis=1)
                    y= keras.utils.to_categorical(y, num_classes=2)
                    X = np.squeeze(X)
                    y = np.squeeze(y)

                    print("index", index, adv_x.shape, ground_truth.shape)

                    lcr, variance, kde = get_uncertain_predictions(detector_model, np.array([adv_x,ground_truth]), ref_x,ref_y)
                    lcr, variance, kde = np.array(lcr).reshape((-1,1)), np.array(variance).reshape((-1,1)), np.array(kde).reshape((-1,1))

                    metrics = np.concatenate((lcr, variance, kde),axis=1)

                    #print("****LCR",lcr.shape,variance.shape,kde.shape,metrics.shape)
                    for i,x in enumerate(X):
                        yield x, y[i], metrics[i]
