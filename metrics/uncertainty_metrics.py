import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import sys
sys.path.append("./")
sys.path.append("./detecting-adversarial-samples/")

import numpy as np
from sklearn.neighbors import KernelDensity

from detect.util import get_noisy_samples, get_mc_predictions, get_deep_representations, score_samples, normalize as normalize_value



# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00}
kernel_densities = {}

def init_kernel_densities(model, X_train, Y_train, batch_size=256, dataset="cifar"):
    class_inds = {}
    global kernel_densities
    kernel_densities = {}
    for i in range(Y_train.shape[1]):
        class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]
        
    X_train_features = get_deep_representations(model, X_train,batch_size=batch_size)
    for i in range(Y_train.shape[1]):
        kernel_densities[i] = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTHS[dataset]).fit(X_train_features[class_inds[i]])

def eval_kernel_density(model, X, batch_size=256, reset_kde=False, **args):
    if reset_kde or len(kernel_densities.keys()) ==0:
        init_kernel_densities(model, batch_size=batch_size, **args)

    features = get_deep_representations(model, X,batch_size=batch_size)
    preds_class =  np.argmax(model.predict(X, verbose=0,batch_size=batch_size), axis=1)

    densities = score_samples(kernel_densities,features,preds_class)

    return densities

def format_lcr(x,ref, ratio=True):
    x = np.argmax(x, axis=3)
    ref = np.argmax(ref, axis=2)
    LCR = []
    for i in range(x.shape[1]):
        preds = x[:,i,:]
        nb_preds = preds.shape[0]
        arg = preds
        diff =  arg-ref
        _shape = diff.shape
        lcr = np.abs(np.reshape(diff, -1))
        lcr = np.minimum([1]*lcr.size,lcr)
        lcr = np.reshape(lcr, _shape)
        lcr = np.sum(lcr,axis=0)
    
        if ratio:
            lcr = lcr/nb_preds
        LCR.append(lcr)

    return  np.array(LCR)

def get_uncertain_predictions(model, inputs, X, Y, nb_dropout_mutants=50, dataset="cifar"):
    inputs_split_shape = inputs.shape

    args = dict(dataset=dataset, X_train=X, Y_train=Y)
    kde = [eval_kernel_density(model, inputs[i,:,:,:,:], reset_kde=True, **args) for i in range(inputs.shape[0])]
    
    inputs_reshaped = np.copy(np.reshape(inputs, (-1,inputs_split_shape[2],inputs_split_shape[3],inputs_split_shape[4])))
    preds = get_mc_predictions(model, inputs_reshaped, nb_iter=nb_dropout_mutants)
    preds = np.reshape(preds, (nb_dropout_mutants, inputs_split_shape[0], inputs_split_shape[1], -1))
    ref = np.repeat(np.array([Y]),nb_dropout_mutants,axis=0)
    lcr = format_lcr(preds, ref)

    x = np.swapaxes(preds,0,1)
    var = x.var(axis=1)
    var_ = var.mean(axis=2) 


    return lcr, var_, kde



if __name__ == "__main__":

  from utils.adversarial_models import load_model
  from attacks import craft_attack

  model, x_train, x_test, y_train, y_test = load_model(dataset="cifar10",model_type="basic",epochs=25)
  x,y  = x_test[:256], y_test[:256]

  #keep only correctly predicted inputs
  batch_size = 64
  preds_test = model.predict_classes(x, verbose=0,batch_size=batch_size)
  inds_correct = np.where(preds_test == y.argmax(axis=1))[0]
  x, y = x[inds_correct], y[inds_correct]

  fgsm_x = np.array(craft_attack(model, x,"fgsm"))
  pgd_x = np.array(craft_attack(model, x,"pgd"))

  lcr, variance, kde = get_uncertain_predictions(model, np.array([fgsm_x,pgd_x]), x,y)
  print()