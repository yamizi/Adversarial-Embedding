from metrics.toolbox_metrics import empirical_robustness, get_crafter
#from utils.attack_classifier import KerasClassifier
from art.classifiers import KerasClassifier

def get_attack_params(attack_name, norm=2, eps=1., minimal=True):
    attack_params = {"norm": norm,'minimal': minimal,"targeted":False}

    if attack_name[:9]=="targeted_":
        attack_params["targeted"] = True
        attack_name=attack_name[9:]

    if attack_name == "pgd":
        attack_params["eps_step"] = 0.1

    if attack_name == "pgd" or attack_name == "fgsm" or attack_name == "bim":
        attack_params["eps"] = eps
    return attack_params, attack_name


def eval_attack_robustness(keras_model, attack_name, norm, x, return_adv=False, adv_x=None, eps=1.):

    attack_params, attack_name = get_attack_params(attack_name, norm, eps)
    classifier = KerasClassifier(model=keras_model)
    return empirical_robustness(classifier, x, attack_name, attack_params, return_adv, adv_x)


def craft_attack(model, x, attack_name, norm=2,y=None, epsilon=1., minimal=True):

    attack_params, attack_name = get_attack_params(attack_name, norm,epsilon, minimal)
    classifier = KerasClassifier(model=model)
    crafter = get_crafter(classifier, attack_name, attack_params)

    adv_x = crafter.generate(x,y)
    return adv_x
