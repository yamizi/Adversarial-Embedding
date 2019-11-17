import sys
sys.path.append("./")
sys.path.append("./detecting-adversarial-samples/")

import numpy as np
from metrics.attacks import eval_attack_robustness
from metrics.toolbox_metrics import clever_u
from utils.adversarial_models import load_model, KerasClassifier


def clever_robustness(model, x_set, norm=2):

    r_l1 = 40
    r_l2 = 2
    r_li = 0.1
    pool_factor=3
    nb_batches = 10
    batch_size = 5
    radius = r_l1 if norm==1 else (r_l2 if norm==2 else r_li)

    keras_model = KerasClassifier(model=model, clip_values=(0, 255))
    scores = []
    for element in x_set:
        score = clever_u(keras_model, element, nb_batches, batch_size, radius, norm=norm, pool_factor=pool_factor)
        scores.append(score)

    return np.min(scores)


def or_robustness(model, x, attack_name=None, perturbation=None, adv_x = None):
    classifier = KerasClassifier(model=model)

    if perturbation is None:
        perturbation, adv_x = eval_attack_robustness(classifier, attack_name, 2, x, True)

    y = classifier.predict(x)
    adv_y = classifier.predict(adv_x)


    # Measure PC
    preds = np.argmax(y, axis=1)
    adv_preds = np.argmax(adv_y, axis=1)
    diff_preds = [dict(true_before=y[i][preds[i]], true_after=adv_y[i][preds[i]], false_before=y[i][adv_preds[i]], false_after=adv_y[i][adv_preds[i]]) for i in range(len(preds))]

    c = 0.000001
    PC = np.array([diff_preds[i]["true_before"]-diff_preds[i]["true_after"]+diff_preds[i]["false_after"]-diff_preds[i]["false_before"] for i in range(len(preds))]) + c
    DP = np.array([diff_preds[i]["false_after"]-diff_preds[i]["true_after"] for i in range(len(preds))]) + c

    or_metrics = PC/DP*perturbation
    or_metric = np.min(or_metrics)

    return or_metric, perturbation, adv_x
    

if __name__ == "__main__":
    model, x_train, x_test, y_train, y_test = load_model(dataset="cifar10",model_type="basic",epochs=5)
    pgd = or_robustness(model, x_test[:128], "pgd")
    cw = or_robustness(model, x_test[:128], "cw")
    fgsm = or_robustness(model, x_test[:128], "fgsm")
    print()