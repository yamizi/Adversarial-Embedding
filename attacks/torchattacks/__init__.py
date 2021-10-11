
from attacks.torchattacks.mifgsm import MIFGSM
from attacks.torchattacks.onepixel import OnePixel
from attacks.torchattacks.cw import CW
from attacks.torchattacks.apgdt import APGDT

from torch import nn
import torch
from attacks import MultiLabelCrossEntropyLoss, MultiLabelHingeLoss, RankLoss, SortedLoss, LInfLoss, TopkLoss,\
    TopkSortedLoss, WarpLoss, TopkLossV2, TopkLossV3, TopkSortedLossSampled, OrdinalLoss

def generate(parameters, model,x_test,y_target , min_pixel_value, max_pixel_value, min_val, max_val, capacity,
             use_cuda=True, experiment=None, data_size = None, strategy="binary", batch_index=0):


    reduction = parameters.get("reduction","mean")
    criterion_val = parameters.get("criterion")
    criterion_val = criterion_val.split("_")

    criterion_variant = criterion_val[1] if len(criterion_val)==2 else None
    criterion_val = criterion_val[0]

    if criterion_val == "mse":
        criterion = nn.MSELoss(reduction=reduction)
    elif criterion_val == "mae":
            criterion = nn.L1Loss(reduction=reduction)
    elif criterion_val == "bce":
        criterion = nn.BCEWithLogitsLoss(reduction=reduction)
    elif criterion_val == "mce":
        criterion = MultiLabelCrossEntropyLoss(reduction=reduction)
    elif criterion_val == "mhl":
        criterion = MultiLabelHingeLoss(reduction=reduction)
    elif criterion_val == "ce":
        criterion = nn.CrossEntropyLoss(reduction=reduction)
    elif criterion_val == "kl":
        criterion = nn.KLDivLoss(reduction=reduction)
    elif criterion_val == "hl":
        criterion = nn.MultiLabelMarginLoss(reduction=reduction)

    elif criterion_val == "nll":
        criterion = nn.NLLLoss(reduction=reduction)
    elif criterion_val == "hel":
        criterion = nn.HingeEmbeddingLoss(reduction=reduction)

    elif criterion_val == "rl":
        criterion = RankLoss(reduction=reduction)

    elif criterion_val == "sl":
        criterion = SortedLoss(reduction=reduction)

    elif criterion_val == "linf":
        criterion = LInfLoss(reduction=reduction)

    elif criterion_val == "warp":
        criterion = WarpLoss(reduction=reduction)

    else:
        criterion = None

    if parameters.get("strategy") == "sorted":

        if criterion_variant is None:
            criterion = TopkLoss(reduction=reduction, criterion=criterion)

        elif criterion_variant == "v2":
            criterion = TopkLossV2(reduction=reduction, criterion=criterion, experiment=experiment)

        elif criterion_variant == "v3":
            criterion = TopkLossV3(reduction=reduction, criterion=criterion, experiment=experiment)

        elif criterion_variant == "vN":
            criterion = TopkSortedLoss(reduction=reduction, criterion=criterion, experiment=experiment)

        elif criterion_variant == "vS":
            criterion = TopkSortedLossSampled(reduction=reduction, criterion=criterion, experiment=experiment)

        elif criterion_variant == "ord":
            criterion = OrdinalLoss(reduction=reduction, criterion=criterion, experiment=experiment)

        elif criterion_variant == "vSLS":
            criterion = TopkSortedLossSampled(reduction=reduction, criterion=criterion, experiment=experiment, log_sum=True)


    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    x_test = x_test.to(device)
    y_target = y_target.to(device)

    if hasattr(criterion,"to"):
        criterion = criterion.to(device)

    if parameters.get("algorithm") == "mifgsm":
        attack = MIFGSM(model,parameters.get("max_eps"),parameters.get("max_iter"), loss=criterion,
                        threshold=(min_val+max_val)/2, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value,
                        experiment=experiment, data_size=data_size, strategy=strategy, random_noise=parameters.get("random_noise"), batch_index=batch_index)

    if parameters.get("algorithm") == "apgdt":
        attack = APGDT(model,eps=parameters.get("max_eps"),steps = parameters.get("max_iter"),verbose=True,
                       n_classes=parameters.get("model_capacity"), threshold=(min_val+max_val)/2,
                       min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value,experiment=experiment,
                       data_size=data_size, strategy=strategy, random_noise=parameters.get("random_noise"))

    if parameters.get("algorithm") == "pixel":
        attack = OnePixel(model,parameters.get("max_eps"),parameters.get("max_iter"), threshold=(min_val+max_val)/2,
                          experiment=experiment, loss=criterion, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value)

    if parameters.get("algorithm") == "cw":
        attack = CW(model,parameters.get("max_eps"),parameters.get("max_iter"), loss=criterion, threshold=(min_val+max_val)/2, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value)

    if parameters.get("algorithm") != "apgdt":
        attack.set_mode_targeted()

    x_test_adv= attack(x_test,y_target)

    return x_test_adv

