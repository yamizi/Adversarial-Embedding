#From https://github.com/columbia/MTRobust

from torch.autograd import Variable
import torch
import numpy as np
import math

def clamp_tensor(image, upper_bound, lower_bound):
    image = torch.where(image > upper_bound, upper_bound, image)
    image = torch.where(image < lower_bound, lower_bound, image)
    return image

def back_transform(image, info):
    # image = image2.copy()

    image[:, 0, :, :] = image[:, 0, :, :] * info["std"][0] + info["mean"][0]
    image[:, 1, :, :] = image[:, 1, :, :] * info["std"][1] + info["mean"][1]
    image[:, 2, :, :] = image[:, 2, :, :] * info["std"][2] + info["mean"][2]
    return image

def forward_transform(image, info):
    image[:, 0, :, :] = (image[:, 0, :, :] - info["mean"][0]) / info["std"][0]
    image[:, 1, :, :] = (image[:, 1, :, :] - info["mean"][1]) / info["std"][1]
    image[:, 2, :, :] = (image[:, 2, :, :] - info["mean"][2]) / info["std"][2]
    return image

class Houdini(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Y_pred, Y, task_loss, ignore_index=255):

        normal_dist     = torch.distributions.Normal(0.0, 1.0)
        probs           = 1.0 - normal_dist.cdf(Y_pred - Y)
        loss            = torch.sum(probs * task_loss.squeeze()) #* mask.squeeze(1)) / torch.sum(mask.float())

        ctx.save_for_backward(Y_pred, Y, task_loss)
        return loss

    @staticmethod
    def backward(ctx, grad_output):

        Y_pred, Y, task_loss = ctx.saved_tensors

        C = 1./math.sqrt(2 * math.pi)

        grad_input  = C * torch.exp(-1.0 * (torch.abs(Y - Y_pred) ** 2) / 2.0) * task_loss.squeeze()

        return (grad_output * grad_input, None, None, None)


"""
class Houdini(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Y_pred, Y, task_loss, ignore_index=255):

        max_preds, max_inds = Y_pred.max(axis=1)

        mask            = (Y != ignore_index)
        Y               = torch.where(mask, Y, torch.zeros_like(Y).to(Y.device))
        true_preds      = torch.gather(Y_pred, 1, Y).squeeze(1)

        normal_dist     = torch.distributions.Normal(0.0, 1.0)
        probs           = 1.0 - normal_dist.cdf(true_preds - max_preds)
        loss            = torch.sum(probs * task_loss.squeeze(1) * mask.squeeze(1)) / torch.sum(mask.float())

        ctx.save_for_backward(Y_pred, Y, mask, max_preds, max_inds, true_preds, task_loss)
        return loss

    @staticmethod
    def backward(ctx, grad_output):

        Y_pred, Y, mask, max_preds, max_inds, true_preds, task_loss = ctx.saved_tensors

        C = 1./math.sqrt(2 * math.pi)

        temp        = C * torch.exp(-1.0 * (torch.abs(true_preds - max_preds) ** 2) / 2.0) * task_loss.squeeze(1) * mask.squeeze(1)
        grad_input  = torch.zeros_like(Y_pred).to(Y_pred.device)

        grad_input.scatter_(1, max_inds.unsqueeze(1), temp.unsqueeze(1))
        grad_input.scatter_(1, Y, -1.0 * temp.unsqueeze(1))

        grad_input          /= torch.sum(mask.float())

        return (grad_output * grad_input, None, None, None)
"""
