"""
Cyoher-text indistinguishibility
https://en.wikipedia.org/wiki/Ciphertext_indistinguishability
"""

import sys
sys.path.append("./")
from experiments import logger, RANDOM_SEED, DATASET_CLASSES

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from keras.utils import to_categorical


def ind_cpa():

    """
    -The challenger generates a key pair PK, SK based on some security parameter k (e.g., a key size in bits), 
    and publishes PK to the adversary. The challenger retains SK.
    -The adversary may perform a polynomially bounded number of encryptions or other operations.
    -Eventually, the adversary submits two distinct chosen plaintexts M 0 , M 1 {\displaystyle \scriptstyle M_{0},M_{1}} 
    \scriptstyle M_{0},M_{1} to the challenger.
    The challenger selects a bit b ∈ {\displaystyle \scriptstyle \in } \scriptstyle \in {0, 1} uniformly at random, and 
    sends the challenge ciphertext C = E(PK, M b {\displaystyle \scriptstyle M_{b}} \scriptstyle M_{b}) back to the adversary.
    The adversary is free to perform any number of additional computations or encryptions. Finally, it outputs a guess for the 
    value of b.

    Our case

    
    """

def run(dataset="cifar10",model_type="basic", epochs = 25, exp_id="_gen_dataset"):
    ind_cpa()


    
    
if __name__ == "__main__":
    run(model_type="basic")