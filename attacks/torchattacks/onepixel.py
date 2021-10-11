import numpy as np

import torch
from attacks.torchattacks.base import BaseAttack
from attacks import success_strategy
from torchattacks.attacks._differential_evolution import differential_evolution
from utils.rs import ssim

class OnePixel(BaseAttack):
    r"""
    Attack in the paper 'One pixel attack for fooling deep neural networks'
    [https://arxiv.org/abs/1710.08864]
    
    Modified from "https://github.com/DebangLi/one-pixel-attack-pytorch/" and 
    "https://github.com/sarathknv/adversarial-examples-pytorch/blob/master/one_pixel_attack/"
    
    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        pixels (int): number of pixels to change (DEFAULT: 1)
        steps (int): number of steps. (DEFAULT: 75)
        popsize (int): population size, i.e. the number of candidate agents or "parents" in differential evolution (DEFAULT: 400)
        inf_batch (int): maximum batch size during inference (DEFAULT: 128)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.OnePixel(model, pixels=1, steps=75, popsize=400, inf_batch=128)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, pixels=1, steps=75, popsize=400, inf_batch=128, threshold:float=0.5, loss=None, min_pixel_value=0, max_pixel_value=1, experiment=None, data_size=None, random_weight=0.01,random_noise="fading",
                 strategy="binary", batch_index=0):

        super(OnePixel, self).__init__("OnePixel", model, steps=steps, min_pixel_value=min_pixel_value,
                                     max_pixel_value=max_pixel_value,
                                     threshold=threshold, experiment=experiment, data_size=data_size,
                                     random_weight=random_weight, random_noise=random_noise,
                                     strategy=strategy)

        self.pixels = pixels
        self.steps = steps
        self.popsize = popsize
        self.inf_batch = inf_batch
        self.threshold = threshold
        self.min_pixel_value = min_pixel_value
        self.max_pixel_value=max_pixel_value
        self.loss = loss
        self.batch_index = batch_index
        self.input_index = 0


    def rescale(self, images):
        return (images-self.min_pixel_value) / self.max_pixel_value

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)
        
        batch_size, channel, height, width = images.shape
        
        bounds = [(0, height), (0, width)]+[(self.min_pixel_value, self.max_pixel_value)]*channel
        bounds = bounds*self.pixels
        
        popmul = max(1, int(self.popsize/len(bounds)))
                     
        adv_images = []
        for idx in range(batch_size):
            self.input_index = idx
            self.input_steps = 0
            image, label = images[idx:idx+1].cpu(), labels[idx:idx+1].cpu()
            delta = differential_evolution(func=lambda delta: self._loss(image, label, delta),
                                           bounds=bounds,
                                           callback=lambda delta, convergence:\
                                                     self._attack_success(image, label, delta),
                                           maxiter=self.steps, popsize=popmul,
                                           init='random',
                                           recombination=1, atol=-1, 
                                           polish=False).x
            delta = np.split(delta, len(delta)/len(bounds))
            adv_image = self._perturb(image, delta)

            data_size = self.data_size if self.data_size is not None else label.shape[1]
            prob = self._get_prob(adv_image, return_tensor=True)



            success_data = success_strategy(label, prob, data_size=data_size, strategy=self.strategy,
                                            threshold=self.threshold)

            adv_images.append(adv_image)
        
        adv_images = torch.cat(adv_images)
        return adv_images
    
    def _loss(self, image, label, delta):
        adv_images = self._perturb(image, delta)  # Mutiple delta

        if self.loss:
            output = self._get_prob(adv_images, True)
            loss = self.loss(output, label.cuda().repeat_interleave(adv_images.shape[0], axis=0)).cpu().detach().numpy()
            error = loss.mean(1)
        else:
            label = label.numpy()
            l = label>self.threshold
            prob = self._get_prob(adv_images)#[:, label]
            l = l.repeat(prob.shape[0], axis=0)
            p = prob>self.threshold
            loss =  l.astype(int) - p.astype(int)

            error = np.mean(loss, axis=1)

        #print(error.mean())
        return error
    
    def _attack_success(self, image, label, delta):

        adv_image = self._perturb(image, delta) # Single delta

        data_size= self.data_size if self.data_size is not None else label.shape[1]
        prob = self._get_prob(adv_image, return_tensor=True)

        loss = self._loss(image, label, delta)
        success_data = success_strategy(label, prob, data_size=data_size, strategy=self.strategy, threshold=self.threshold)

        print(image.shape,adv_image.shape)
        sim = ssim(image.cuda(),adv_image.cuda())
        print("success strategy adv", self.input_index , self.input_steps , success_data, "ssim", sim)

        self.plot_metrics(success_data, self.input_steps, None, cost=loss[0], outputs=prob,labels=label, epoch=self.input_index)
        self.input_steps+=1

        if (self._targeted == 1) and (success_data == 1):
            return True
        elif (self._targeted == -1) and (success_data == 0):
            return True
        return False


        lbl = label.numpy()<self.threshold
        pre = prob<self.threshold
        #print("succes",(pre == lbl).sum()/pre.size)
        if (self._targeted == 1) and (pre == lbl).all():
            return True
        elif (self._targeted == -1) and (pre != lbl).all():
            return True
        return False
    
    def _get_prob(self, images, return_tensor=False):
        with torch.no_grad():
            batches = torch.split(images, self.inf_batch)
            outs = []
            for batch in batches:
                out = self.model(batch)
                outs.append(out)
        outs = torch.cat(outs)
        prob = outs #F.softmax(outs, dim=1)
        return prob if return_tensor else prob.detach().cpu().numpy()
    
    def _perturb(self, image, delta):
        delta = np.array(delta)
        if len(delta.shape) < 2:
            delta = np.array([delta])
        num_delta = len(delta)
        adv_image = image.clone().detach().to(self.device)
        adv_images = torch.cat([adv_image]*num_delta, dim=0)
        for idx in range(num_delta):
            pixel_info = delta[idx].reshape(self.pixels, -1)
            for pixel in pixel_info:
                pos_x, pos_y = pixel[:2]
                channel_v = pixel[2:]
                for channel, v in enumerate(channel_v):
                    adv_images[idx, channel, int(pos_x), int(pos_y)] = v
        return adv_images
