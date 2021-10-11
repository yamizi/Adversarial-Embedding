"""
Module providing evasion attacks under a common interface.
"""
import torch
import math
from math import exp, log
from differentiable_sorting.torch import bitonic_matrices, diff_argsort
import numpy as np
from torch import optim, nn
from attacks.warp import WARPLoss
from attacks.houdini_loss import Houdini

from torch.nn import Parameter as P
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import functional as F


def WarpLoss(reduction="mean"):
    wl = WARPLoss()

    def loss(y_pred, y_true):
        y_target = (y_true > 0).float()
        l = wl(y_pred, y_target)
        return l

    return loss


def MultiLabelCrossEntropyLoss(reduction="mean"):
    def multilabel_categorical_crossentropy(y_pred, y_true):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        if reduction == "mean":
            return (neg_loss + pos_loss).mean()
        else:
            return neg_loss + pos_loss

    return multilabel_categorical_crossentropy


def MultiLabelCrossEntropyLoss(reduction="mean"):
    def multilabel_categorical_crossentropy(y_pred, y_true):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        if reduction == "mean":
            return (neg_loss + pos_loss).mean()
        else:
            return neg_loss + pos_loss

    return multilabel_categorical_crossentropy


def MultiLabelHingeLoss(reduction="mean"):
    def multilabel_hinge(y_pred, y_true):

        loss = y_true * (1 - y_pred) + (1 - y_true) * y_pred
        if reduction == "mean":
            return loss.mean()
        else:
            return loss

    return multilabel_hinge


def LInfLoss(reduction="mean"):
    def loss_fn(y_pred, y_true):
        loss = torch.cdist(y_true, y_pred, np.inf)
        return loss.mean() if reduction == "mean" else loss

    return loss_fn


def RankLoss(reduction="mean"):
    def rank(y_pred, y_target_binary):
        y_target_binary = y_target_binary.flatten()
        y_pred = y_pred.flatten()
        pos_args = y_target_binary.round() == 1
        neg_args = y_target_binary.round() == 0

        loss = torch.max(torch.exp(y_pred[neg_args])) - torch.min(torch.exp(y_pred[pos_args]))
        return torch.clamp(loss, min=0)

    return rank


def TopkLoss(reduction="mean", criterion=None):
    if criterion is None:
        criterion = nn.MSELoss(reduction=reduction)

    def loss(y_pred, y_target):
        k = (y_target[0] > 0).sum().item()
        arg_target = torch.argsort(-y_target)[:, :k]
        val_targets = y_target[:, arg_target]
        val_pred = y_pred[:, arg_target]
        loss1 = criterion(val_targets.float(), val_pred.float())

        arg_pred = torch.argsort(-y_pred)[:, :k]
        val_targets = y_target[:, arg_pred]
        val_pred = y_pred[:, arg_pred]
        loss2 = criterion(val_targets.float(), val_pred.float())

        loss3 = criterion(y_pred, y_target)
        l = loss1 + loss2  # + loss3

        return l

    return loss


def TopkLossV2(reduction="mean", criterion=None, experiment=None):
    if criterion is None:
        criterion = nn.MSELoss(reduction=reduction)

    wl = WARPLoss()
    sample_loss = TopkSortedLossSampled(experiment=experiment)

    def loss(y_pred, y_target):
        k = (y_target[0] > 0).sum().item()

        # top k true labels probabilities match their predictions probabilities
        arg_target = torch.argsort(-y_target)[:, :k]
        val_targets = y_target[:, arg_target]
        val_pred = y_pred[:, arg_target]
        # loss1 = criterion(val_targets.float(), val_pred.float())
        loss1 = torch.max(val_targets.float() - val_pred.float())

        # labels of top k prediction have probabilities that match these labels probabilities
        arg_pred = torch.argsort(-y_pred)[:, :k]
        val_targets = y_target[:, arg_pred]
        val_pred = y_pred[:, arg_pred]
        # loss2 = criterion(val_targets.float(), val_pred.float())
        loss2 = torch.max(val_targets.float() - val_pred.float())

        y_true = (y_target > 0).float()
        l = (loss1 + loss2)

        if experiment is not None:
            experiment.log_metric("loss1", loss1)
            experiment.log_metric("loss2", loss2)

            # sample_loss(y_pred, y_target)
            # loss3 = wl(y_pred, y_true)
            # experiment.log_metric("loss3", loss3)

        return l

    return loss


def TopkLossV3(reduction="mean", criterion=None, experiment=None):
    if criterion is None:
        criterion = nn.MSELoss(reduction=reduction)

    wl = WARPLoss()
    sample_loss = TopkSortedLossSampled(experiment=experiment)

    def loss(y_pred, y_target):
        k = (y_target[0] > 0).sum().item()

        # top k true labels probabilities match their predictions probabilities
        arg_target = torch.argsort(-y_target)[:, :k]
        val_targets = y_target[:, arg_target]
        val_pred = y_pred[:, arg_target]
        # loss1 = torch.linalg.norm(val_targets.float()-val_pred.float())
        loss1 = criterion(torch.exp(val_targets.float()), torch.exp(val_pred.float()))

        # labels of top k prediction have probabilities that match these labels probabilities
        arg_pred = torch.argsort(-y_pred)[:, :k]
        val_targets = y_target[:, arg_pred]
        val_pred = y_pred[:, arg_pred]
        loss2 = criterion(torch.exp(val_targets.float()), torch.exp(val_pred.float()))

        y_true = (y_target > 0).float()

        # sample_loss(y_pred,y_target)
        l = (loss1 + loss2)

        if experiment is not None:
            experiment.log_metric("loss1", loss1)
            experiment.log_metric("loss2", loss2)

            # loss3 = wl(y_pred, y_true)
            # experiment.log_metric("loss3", loss3)

        return l

    return loss


def TopkSortedLoss(reduction="mean", criterion=None, experiment=None):
    if criterion is None:
        criterion = nn.MSELoss(reduction=reduction)

    def loss(y_pred, y_target):
        ref_target = y_target[0]
        nb_labels = len(ref_target)
        alpha = 0
        l = 0

        for k in range(nb_labels):
            mask_j = ref_target < ref_target[k]

            for j in range(nb_labels):
                if mask_j[j]:
                    l = l + torch.max(torch.Tensor([0, alpha + y_pred[:, j] - y_pred[:, k]]))
        return l

    return loss


def TopkSortedLossSampled(reduction="mean", criterion=None, random_sample=50, experiment=None, log_sum=False):
    houdini = Houdini()
    if criterion is None:
        criterion = nn.MSELoss(reduction=reduction)

    def loss(y_pred, y_target):
        ref_labels = y_target[0]
        ref_targets = torch.nonzero(ref_labels)
        nb_labels = ref_labels.shape[0]
        alpha = 0.05
        l = 0

        for k in ref_targets:
            mask_j = ref_labels < ref_labels[k.item()]
            counter = random_sample
            while counter > 0:
                j = np.random.choice(range(nb_labels), 1)
                if mask_j[j]:
                    if log_sum:
                        l = l + torch.exp(criterion(y_pred[:, j.item()], y_pred[:, k.item()]))
                    else:
                        l = l + torch.max(
                            torch.Tensor([0, alpha + criterion(y_pred[:, j.item()], y_pred[:, k.item()])]))
                    counter = counter - 1

        if experiment is not None:
            experiment.log_metric("loss4_sampled", l)

        if log_sum:
            l = torch.log(1 + l)
        l = Houdini.apply(y_pred, y_target, l)
        return l

    return loss


def SortedLoss(reduction="mean", cpu=True):
    def loss(y_pred, y_target):
        # y_target = y_target.flatten()#.cpu()#[:1024]
        # y_pred = y_pred.flatten()#.cpu()#[:1024]

        power_size = np.ceil(np.log(len(y_target)) / np.log(2))
        missing = int(2 ** power_size - len(y_target))
        y_target = torch.cat([y_target, torch.zeros(missing).cuda()])
        y_pred = torch.cat([y_pred, torch.zeros(missing).cuda()])

        if cpu:
            y_pred = y_pred.cpu()
            y_target = y_target.cpu()

        sort_matrices = bitonic_matrices(len(y_pred), y_pred.device)
        sort_outs = diff_argsort(sort_matrices, -y_pred)
        sort_lbls = diff_argsort(sort_matrices, -y_target)
        return nn.MSELoss(reduction=reduction)(sort_outs.float(), sort_lbls.float())

        arg_sort_outs = torch.argsort(-y_pred, dim=1)
        arg_sort_lbls = torch.argsort(-y_target, dim=1)
        return nn.MSELoss(reduction=reduction)(arg_sort_outs.float(), arg_sort_lbls.float())
        # err = arg_sort_outs != arg_sort_lbls
        # return err.float().mean()

    return loss


def ordinal_loss(input: torch.Tensor, target: torch.Tensor, MaxValue):
    if (torch.argsort(input) == torch.argsort(target)).all():
        return 0
    else:
        in_padL = F.pad(input, [1, 0], mode='constant', value=input[-1].data)
        in_padR = F.pad(input, [0, 1], mode='constant', value=input[0].data)
        in_diff = in_padR - in_padL
        tar_padL = F.pad(target, [1, 0], mode='constant', value=target[-1].data)
        tar_padR = F.pad(target, [0, 1], mode='constant', value=target[0].data)
        tar_diff = tar_padR - tar_padL
        loss = F.mse_loss(in_diff / MaxValue, tar_diff / MaxValue)
        return loss


class OrdinalLoss(_WeightedLoss):
    def __init__(self, MaxValue=1, weight=None, size_average=None, reduce=None, reduction='mean', experiment=None,
                 criterion=None):
        super(OrdinalLoss, self).__init__(weight, size_average, reduce, reduction)
        assert MaxValue != 0
        self.MaxVaule = MaxValue
        self.w = P(torch.Tensor(2))
        self.experiment = experiment
        self.criterion = F.binary_cross_entropy if criterion is None else criterion

    def forward(self, input: torch.Tensor, target: torch.Tensor, include_ce=True):
        l = 0
        for i, trg in enumerate(target):
            l = l + ordinal_loss(F.softmax(input[i],0), F.softmax(trg,0), self.MaxVaule)

        l = l / len(target) if self.reduction == "mean" else l
        if include_ce:
            l = self.criterion(input, target) + l

        else:
            l = l

        if self.experiment is not None:
            self.experiment.log_metric("loss_ordinal", l)
        return l


def success_strategy(labels, outputs, strategy="binary", threshold=0.5, data_size=None):
    success = None
    if data_size is not None:
        outs = outputs[:, :data_size]
        lbls = labels[:, :data_size]
    else:
        lbls = labels
        outs = outputs

    outs = outs if isinstance(outs, np.ndarray) else outs.cpu().detach().numpy()
    lbls = lbls if isinstance(lbls, np.ndarray) else lbls.cpu().detach().numpy()

    if strategy == "binary":
        outs = outs > threshold
        lbls = lbls > threshold
        success = (outs == lbls).sum(axis=1) / lbls.shape[1]

    elif strategy == "decimal":
        outs = outs.round(decimals=1)
        lbls = lbls.round(decimals=1)
        success = (outs == lbls).sum(axis=1) / lbls.shape[1]

    elif strategy == "sorted":
        nb_non_null = (lbls > 0).sum(axis=1)[0]
        arg_sort_outs = np.argsort(-outs, axis=1)
        arg_sort_lbls = np.argsort(-lbls, axis=1)
        equal = arg_sort_outs[:, :nb_non_null] == arg_sort_lbls[:, :nb_non_null]
        success = equal.sum(axis=1) / nb_non_null

    return success


def make_one_hot(labels, num_classes, device):
    '''
    Converts an integer label to a one-hot values.
    Parameters
    ----------
        labels : N x H x W, where N is batch size.(torch.Tensor)
        num_classes : int
        device: torch.device information
    -------
    Returns
        target : torch.Tensor on given device
        N x C x H x W, where C is class number. One-hot encoded.
    '''

    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    one_hot = one_hot.to(device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def prepare_message(message, capacity, input_shape, parameters, bucket_size=1, min_val=0, max_val=1, padding=True):
    if parameters.get("dense", False):
        return prepare_dense_message(message, capacity, input_shape, parameters)
    else:
        pixels = input_shape[1] * input_shape[2]
        return prepare_classification_message(message, capacity, pixels, parameters, bucket_size, min_val, max_val,
                                              padding)


def prepare_dense_message(message, capacity, input_shape, parameters):
    msg = ''.join(format(x, 'b') for x in bytearray(str(message), 'utf-8'))

    output_shape = (parameters.get("output_size", 5), parameters.get("output_size", 5))
    encoded = np.zeros(output_shape)
    absolute_capacity = message_size = len(msg)
    density = log(capacity) / log(2)
    bucket_size = int(math.floor(density))

    if message_size > bucket_size * np.prod(output_shape):
        print("Message larger than our steganographer capacity; please change the DNN steganographer")
        return None, absolute_capacity, density

    density = bucket_size * np.prod(output_shape) / (input_shape[1] * input_shape[2])

    for i in range(0, message_size, bucket_size):
        bucket_msg = msg[i:i + bucket_size]
        decimal_msg = "{}".format(int(bucket_msg, 2))
        j = (i // bucket_size) // output_shape[0]
        k = (i // bucket_size) % output_shape[0]
        encoded[j, k] = decimal_msg

    return encoded, absolute_capacity, density


def prepare_classification_message(message, capacity, pixels, parameters, bucket_size=1, min_val=0, max_val=1,
                                   padding=True):
    strategy = parameters.get("strategy", "binary")
    msg = ''.join(format(x, 'b') for x in bytearray(str(message), 'utf-8'))
    absolute_capacity = len(msg)
    density = absolute_capacity / pixels
    print("Message size {} bits, or {} bpp".format(absolute_capacity, density))

    if strategy == "binary":
        encoded = [msg[i:i + bucket_size] for i in range(0, len(msg), bucket_size)]
        if len(encoded) > capacity:
            print("Message larger than our steganographer capacity; please change the DNN steganographer")
            return None, absolute_capacity, density

        if padding:
            encoded = np.array(
                list(np.array(encoded, int) * (max_val - min_val) + min_val) + [min_val] * (capacity - len(encoded)))
        else:
            encoded = np.array(
                list(np.array(encoded, int) * (max_val - min_val) + min_val))

    elif strategy == "decimal":
        decimal_msg = "{}".format(int(msg, 2))
        encoded = [decimal_msg[i:i + bucket_size] for i in range(0, len(decimal_msg), bucket_size)]

        if padding:
            encoded = encoded + [0] * (capacity - len(encoded))

        encoded = np.array(encoded, int) / 10
        absolute_capacity = len(encoded)


    elif strategy == "sorted":
        output = [0] * capacity
        chunk_size = math.floor(log(capacity) / log(2))
        decomposition_power = 2
        minimum_proba = parameters.get("minimum_proba", 0)

        ordered = [[]]

        if isinstance(message, list):
            ordered = message
        else:
            for i in range(0, len(msg), chunk_size):
                chunk = msg[i:i + chunk_size]
                decimal_chunk = int(chunk, 2)
                if decimal_chunk in ordered[-1]:
                    ordered.append([decimal_chunk])
                else:
                    ordered[-1].append(decimal_chunk)

            if len(ordered) > 1:
                print("Alert! {} images required. classes to encode {}".format(len(ordered), [len(e) for e in ordered]))

            ordered = ordered[0]
        print("classes to encode {} vs total model classes {}".format(ordered, capacity))
        for i, index in enumerate(ordered):
            threshold = 1 / len(ordered)
            output[index] = (len(ordered) - i) * threshold * (
                        1 - minimum_proba) + minimum_proba  # max(0, math.pow(1 - i * threshold, power))

        encoded = np.array(output)
        absolute_capacity = len(ordered)

    return encoded, absolute_capacity, density


def check_dense_success(prediction, labels: np.ndarray, classifier: "CLASSIFIER_TYPE", targeted: bool = False, ):
    p = prediction > (classifier.output_values[0] + classifier.output_values[1]) / 2
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy()
    l = labels > (classifier.output_values[0] + classifier.output_values[1]) / 2
    if targeted:
        attack_success = l == p
    else:
        attack_success = p != l

    attack_success = attack_success.mean(axis=1)

    return attack_success


def compute_success_array(
        classifier: "CLASSIFIER_TYPE",
        x_clean: np.ndarray,
        labels: np.ndarray,
        x_adv: np.ndarray,
        targeted: bool = False,
        batch_size: int = 1,
) -> float:
    """
    Compute the success rate of an attack based on clean samples, adversarial samples and targets or correct labels.

    :param classifier: Classifier used for prediction.
    :param x_clean: Original clean samples.
    :param labels: Correct labels of `x_clean` if the attack is untargeted, or target labels of the attack otherwise.
    :param x_adv: Adversarial samples to be evaluated.
    :param targeted: `True` if the attack is targeted. In that case, `labels` are treated as target classes instead of
           correct labels of the clean samples.
    :param batch_size: Batch size.
    :return: Percentage of successful adversarial samples.
    """
    prediction = classifier.predict(x_adv, batch_size=batch_size)

    onehot = labels.sum() == 1
    if onehot:
        adv_preds = np.argmax(prediction, axis=1)
        if targeted:
            attack_success = adv_preds == np.argmax(labels, axis=1)
        else:
            preds = np.argmax(classifier.predict(x_clean, batch_size=batch_size), axis=1)
            attack_success = adv_preds != preds
    else:
        attack_success = check_dense_success(prediction, labels, classifier, targeted)

    return attack_success


def compute_success(
        classifier: "CLASSIFIER_TYPE",
        x_clean: np.ndarray,
        labels: np.ndarray,
        x_adv: np.ndarray,
        targeted: bool = False,
        batch_size: int = 1,
) -> float:
    """
    Compute the success rate of an attack based on clean samples, adversarial samples and targets or correct labels.

    :param classifier: Classifier used for prediction.
    :param x_clean: Original clean samples.
    :param labels: Correct labels of `x_clean` if the attack is untargeted, or target labels of the attack otherwise.
    :param x_adv: Adversarial samples to be evaluated.
    :param targeted: `True` if the attack is targeted. In that case, `labels` are treated as target classes instead of
           correct labels of the clean samples.
    :param batch_size: Batch size.
    :return: Percentage of successful adversarial samples.
    """
    attack_success = compute_success_array(classifier, x_clean, labels, x_adv, targeted, batch_size)
    return np.sum(attack_success) / x_adv.shape[0]
