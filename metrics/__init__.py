import logging
import time
import sys
import numpy as np
from sklearn.neighbors import KernelDensity
from art.classifiers import KerasClassifier
from .attacks import eval_attack_robustness
from .toolbox_metrics import clever_u

# Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_activation_layers(model, x, sorted_layers=True):
    sys.path.append("./keract/")
    from keract import get_activations

    act = get_activations(model, x)

    if not sorted_layers:
        return act.values()
    layers = dict([(k[0:k.index("/")],v) for (k,v) in act.items()])
    layers = [layers.get(y.name) for y in model.layers]

    return layers



class Metrics():
    
    @staticmethod
    def eval_robustness(keras_model, x_set, scores=["clever"]):
        robustness = {}
        if not keras_model:
            return 
        begin_robustness = time.time() 
        try:
            norm = 2
            r_l1 = 40
            r_l2 = 2
            r_li = 0.1
            nb_batches = 10
            batch_size = 5
            radius = r_l1 if norm==1 else (r_l2 if norm==2 else r_li)
            
            keras_model = KerasClassifier(model=keras_model, clip_values=(0, 255))
            
            if "clever" in scores:
                
                scores = []
                for element in x_set:
                    score = clever_u(keras_model, element, nb_batches, batch_size, radius, norm=norm, pool_factor=3)
                    scores.append(score)
                robustness["clever_score"] = np.average(scores)
            if "pgd" in scores:
                robustness["pgd_score"] = eval_attack_robustness(keras_model, "pgd", norm,x_set)
            if "cw" in scores:
                robustness["cw_score"] = eval_attack_robustness(keras_model, "cw", norm,x_set)
            if "fgsm" in scores:
                robustness["fgsm_score"] = eval_attack_robustness(keras_model, "fgsm", norm,x_set)
            
        except Exception as e:
            import traceback
            logger.error("error",e)
            print (traceback.format_exc())
        
        robustness["robustness_time"] = time.time() - begin_robustness
        logger.info('model robustness {}:{}'.format((robustness.keys(), robustness.values())))

        return robustness
