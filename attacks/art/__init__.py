from attacks.art.auto_projected_gradient_descent import AutoProjectedGradientDescent
from attacks.art.projected_gradient_descent_pytorch import PGDMinimal, PGDInterpolated,ProjectedGradientDescentPyTorch
from attacks.art.boundary import  BoundaryAttack
from attacks.art.jsma import SaliencyMapMethod as JSMA
from attacks.art.pytorch_classifier import PyTorchClassifier
import numpy as np
from torch import optim, nn

def generate(parameters, model,x_test,y_target , min_pixel_value, max_pixel_value, min_val, max_val, capacity,
             data_size=None, strategy="binary"):
    if parameters.get("criterion") == "mse":
        criterion = nn.MSELoss()
    elif parameters.get("criterion") == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif parameters.get("criterion") == "ce":
        criterion = nn.CrossEntropyLoss()
    elif parameters.get("criterion") == "kl":
        criterion = nn.KLDivLoss()

    elif parameters.get("criterion") == "hl":
        criterion = nn.MultiLabelMarginLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=x_test.shape[1:],
        nb_classes=capacity,
        output_values=(min_val, max_val)
    )
    if criterion._get_name() == 'BCELoss':
        classifier._reduce_labels = False

    if parameters.get("algorithm") == "pgd":
        attack = ProjectedGradientDescentPyTorch(estimator=classifier, eps=parameters.get("max_eps"),
                                                 eps_step=parameters.get("eps_step"), targeted=True,
                                                 num_random_init=parameters.get("num_random_init"), random_eps=False,
                                                 max_iter=parameters.get("max_iter"),
                                                 success_threshold=parameters.get("success_threshold"))
    elif parameters.get("algorithm") == "mpgd":
        attack = PGDMinimal(estimator=classifier, eps=parameters.get("max_eps"), eps_step=parameters.get("eps_step"),
                            num_random_init=parameters.get("num_random_init"), max_iter=parameters.get("max_iter"),
                            success_threshold=parameters.get("success_threshold"))

    elif parameters.get("algorithm") == "ipgd":
        attack = PGDInterpolated(estimator=classifier, eps=parameters.get("max_eps"),
                                 eps_step=parameters.get("eps_step"),
                                 num_random_init=parameters.get("num_random_init"),
                                 max_iter=parameters.get("max_iter"),
                                 success_threshold=parameters.get("success_threshold"))

    elif parameters.get("algorithm") == "apgd":
        attack = AutoProjectedGradientDescent(estimator=classifier, eps=parameters.get("max_eps"), targeted=True,
                                              max_iter=parameters.get("max_iter"),
                                              nb_random_init=parameters.get("num_random_init"),
                                              success_threshold=parameters.get("success_threshold"))
    elif parameters.get("algorithm") == "jsma":
        attack = JSMA(classifier=classifier, theta=parameters.get("max_eps"),
                                              success_threshold=parameters.get("success_threshold"))


    elif parameters.get("algorithm") == "bound":
        attack = BoundaryAttack(estimator=classifier, epsilon=parameters.get("eps_step"), targeted=True,
                                max_iter=parameters.get("max_iter"), num_trial=parameters.get("num_random_init"),
                                success_threshold=parameters.get("success_threshold"))

    x_test_adv = attack.generate(x=x_test, y=y_target)

    return x_test_adv
