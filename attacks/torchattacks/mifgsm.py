import torch
import torch.nn as nn

from attacks.torchattacks.base import BaseAttack
from art.utils import random_sphere
import numpy as np
from attacks import success_strategy

class MIFGSM(BaseAttack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFAULT: 8/255)
        decay (float): momentum factor. (DEFAULT: 1.0)
        steps (int): number of iterations. (DEFAULT: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.MIFGSM(model, eps=8/255, steps=5, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, steps=5, decay=1.0, loss=None, min_pixel_value=0, max_pixel_value=1,
                 threshold:float=0.5, experiment=None, data_size=None, random_weight=0.01,random_noise="fading",
                 strategy="binary", batch_index=0):
        super(MIFGSM, self).__init__("MIFGSM", model, steps=steps, min_pixel_value=min_pixel_value, max_pixel_value=max_pixel_value,
                 threshold=threshold, experiment=experiment, data_size=data_size, random_weight=random_weight,random_noise=random_noise,
                 strategy=strategy)

        self.eps = eps
        self.steps = int(steps/2)
        self.decay = decay
        self.alpha = self.eps / self.steps
        self.loss = loss
        self.batch_index = batch_index



    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)
        data_size = self.data_size if self.data_size is not None else labels.shape[1]

        loss = self.loss if self.loss is not None else nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()
        self.best_adv_images = None
        self.best_adv_score = 0
        self.max_adv_images = None
        self.max_adv_score = 0

        adv_images_noised = images.clone().detach()

        for i in range(self.steps*2):
            adv_images.requires_grad = True

            outputs_noised = self.model(adv_images_noised)
            outputs = self.model(adv_images)


            success_data = success_strategy(labels,outputs,data_size=data_size, strategy=self.strategy, threshold=self.threshold)
            success_data_noised = success_strategy(labels,outputs_noised,data_size=data_size, strategy=self.strategy, threshold=self.threshold)

            if success_data_noised.mean()>success_data.mean():

                cost = self._targeted * loss(outputs_noised[:, :data_size], labels[:, :data_size])
                grad = torch.autograd.grad(cost, adv_images_noised,
                                           retain_graph=False, create_graph=False)[0]

                adv_images = adv_images_noised
                outputs = outputs_noised
            else:
                cost = self._targeted * loss(outputs[:, :data_size], labels[:, :data_size])
                grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]

            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)

            self.plot_metrics(success_data, i, grad_norm.mean().item(),cost.item(), outputs,labels, adv=adv_images.cpu().detach())

            grad = grad / grad_norm.view([-1] + [1] * (len(grad.shape) - 1))
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() - self.alpha * grad.sign()

            if self.random_noise:

                if self.random_noise =="fading":
                    noise = max((self.steps-i)*self.alpha*self.random_weight, self.alpha*self.random_weight/self.steps)
                elif self.random_noise == "triangular":
                    noise = self.triangle[i%self.triangle_size] * self.random_weight
                elif self.random_noise == "triangular_fading":
                    noise = self.triangle[i%self.triangle_size] * self.random_weight * (2*self.steps-i)/(2*self.steps)

                random_perturbation = random_sphere(adv_images.shape[0], np.product(adv_images[0].shape), noise, self.norm)
                random_perturbation = random_perturbation.reshape(adv_images.shape)
                random_perturbation = torch.from_numpy(random_perturbation).to(self.device, dtype=torch.float)
                adv_images_noised = adv_images+random_perturbation

                if i%self.plotskip==0:
                    random_attack_norm = torch.norm(random_perturbation, p=1, dim=1).mean().item()
                    self.experiment.log_metric("random_attack",random_attack_norm, step=i//self.plotskip, epoch=self.batch_index)
                    #self.experiment.log_metric("momentum_attack",
                                               #torch.norm(self.alpha * grad.sign(), p=1, dim=1).mean().item(), step=i//self.plotskip)

            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=self.min_pixel_value, max=self.max_pixel_value).detach()

            delta_noised = torch.clamp(adv_images_noised - images, min=-self.eps, max=self.eps)
            adv_images_noised = torch.clamp(images + delta_noised, min=self.min_pixel_value, max=self.max_pixel_value).detach()
            adv_images_noised.requires_grad_()

        return self.best_images() if self.max_adv_score>0 else adv_images

