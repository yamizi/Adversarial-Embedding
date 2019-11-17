#Encoding multiple values in one image: Ranked Targetted attack
import sys, random, os
sys.path.append("./")
from experiments import logger, RANDOM_SEED

import numpy as np
from utils.sorted_attack import SATA
from utils.adversarial_models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import json, time, math
from random import shuffle

experiment_time = int(time.time())


def _get_bit_count(n):
    return math.floor(math.log2(n))+1



def run(dataset="cifar10",model_type="basic", epochs = 25, experiment_id="SP8b"):
    

    folder = "./experiments/results/experiment{}".format(experiment_id)
    os.makedirs(folder, exist_ok =True)

    
    if RANDOM_SEED>0:
        random.seed(RANDOM_SEED)

    model, x_train, x_test, y_train, y_test = load_model(
        dataset="cifar10", model_type="basic", epochs=25)

    combined = list(zip(x_test, y_test))
    random.shuffle(combined)
    x_test[:], y_test[:] = zip(*combined)
    
    best_rates = {}
    all_rates = {}
    powers = (2,2.5)
    dataset_classes = 10
    classes = (7,)
    epsilons = (0.5,1,2,3,4,5)
    max_iters = (200,)
    nb_elements = 1000

    for nb_classes in classes:
        best_rate = 0
        best_combination = {} 
        all_combinations = []  
        for power in powers:
            for epsilon in epsilons:
                for max_iter in max_iters:
                    if RANDOM_SEED>0:
                        np.random.seed(RANDOM_SEED)
                    values = list(range(0,9))
                    order = np.array([values[:nb_classes] for i in range(nb_elements) if shuffle(values) or len(values)])
                        
                    #order = np.random.randint(0,dataset_classes,(nb_elements,nb_classes))
                    SATA.power = power
                    adv_x = SATA.craft(model, x_test[:nb_elements],order, epsilon=epsilon, max_iter=max_iter)
                    combination =  {"nb_elements":nb_elements,"max_iter":max_iter, "epsilon":epsilon, "power":power, "rate":SATA.rate_best}
                    all_combinations.append(combination)
                    if SATA.rate_best > best_rate:
                        best_rate = SATA.rate_best
                        best_combination = combination
                    
                    logger.info("class {}, combination {}".format(nb_classes, combination))
                    

        all_rates[nb_classes] = all_combinations
        best_rates[nb_classes] = best_combination

    
    with open("{}/{}.json".format(folder, experiment_time), 'a') as f:
        f.write("{}".format(json.dumps({"all":all_rates,"best":best_rates})))

    logger.info("{}".format(best_rates.items()))


if __name__ == "__main__":
    run(model_type="basic")