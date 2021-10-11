
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np
import torch
from PIL import Image

transformations = {
    "cifar10": {"std":[0.2471, 0.2435, 0.2616],"mean":[0.4914, 0.4822, 0.4465]},
    "imagenet":{"std":[0.229, 0.224, 0.225],"mean":[0.485, 0.456, 0.406]},
    "coco":{"std":[0.229, 0.224, 0.225],"mean":[0.485, 0.456, 0.406]},
    "voc2012":{"std":[0.229, 0.224, 0.225],"mean":[0.485, 0.456, 0.406]},
}

class Identity(torch.nn.Module):
    def __init__(self, activation=False):
        super(Identity, self).__init__()
        self.activation = activation

    def forward(self, x):
        if self.activation=="sigmoid":
            #[0,1]
            return torch.sigmoid(x)*2-1 #(1 + torch.sigmoid(x)) / 2
            #[-1,1]
            #return torch.sigmoid(x)*4-3
        elif self.activation=="relu":
            return torch.relu(x)
        else:
            return x


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, clean_dataset, return_label=False):
        self.base = clean_dataset
        self.return_label = return_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image_orig, label = self.base[idx]
        if self.return_label:
            return image_orig,  label
        else:
            return image_orig


def get_model_dataset(parameters, shuffle=False, device="cuda", return_loader=False):
    model = parameters.get("model")
    dataset = parameters.get("dataset")

    if model.find("imagenet")>-1:
        transform = transforms.Compose([  # [1]
            transforms.Resize(256),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=transformations.get("imagenet").get("mean"),  # [6]
                std=transformations.get("imagenet").get("std")  # [7]
            )])

        if model.find("resnet18")>-1:
            model = models.resnet18(pretrained=parameters.get("pretrained"))
        elif model.find("wide_resnet50") > -1:
            model = models.wide_resnet50_2(pretrained=parameters.get("pretrained"))
        elif model.find("resnet50") > -1:
            model = models.resnet50(pretrained=parameters.get("pretrained"))

    elif model.find("cifar")>-1:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transformations.get("cifar10").get("mean"),
                std=transformations.get("cifar10").get("std")
            )])

        model = torch.hub.load("chenyaofo/pytorch-cifar-models", model, pretrained=True)

    elif model.find("coco")>-1:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transformations.get("coco").get("mean"),
                std=transformations.get("coco").get("std")
            )])

        if model=="coco_v2":
            from utils.msc import MSC
            model = MSC(base=torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained='cocostuff164k',
                                        n_classes=182))
        elif model=="coco_yolov5":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


    elif model.find("voc2012")>-1:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transformations.get("voc2012").get("mean"),
                std=transformations.get("voc2012").get("std")
            )])
        model = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained='voc12', n_classes=21)



    if dataset.find("div2k")>-1:
        from utils.loader import DataLoader
        validation = DataLoader('../SteganoGAN/research/data/div2k/val/', limit=np.inf, shuffle=shuffle,
                            batch_size=parameters.get("batch_size"), transform=transform)

    elif dataset.find("cifar10")>-1:
        from torchvision.datasets import CIFAR10
        from torch.utils.data import DataLoader
        #dataset = CIFAR10(root='data/', download=True, transform=transforms.ToTensor())
        test_dataset = CIFAR10(root='data/', train=False, download=True, transform=transform)

        validation = DataLoader(test_dataset, shuffle=shuffle,
                            batch_size=parameters.get("batch_size"))

    batch_t = None

    for batch in validation:
        batch_t = batch[0]
        break

    if parameters.get("use_hidden")=="sigmoid_last":
        model.add_module("output",Identity(activation="sigmoid"))

    elif parameters.get("use_hidden"):
        model.fc = Identity()
        model.avgpool = Identity(activation="sigmoid")


    model.eval()
    if device=="cuda":
        model = model.cuda()


    if return_loader:
        return validation, model

    return batch_t, model


def back_transform(image, dataset):
    info =transformations.get(dataset)
    image[:, 0, :, :] = image[:, 0, :, :] * info["std"][0] + info["mean"][0]
    image[:, 1, :, :] = image[:, 1, :, :] * info["std"][1] + info["mean"][1]
    image[:, 2, :, :] = image[:, 2, :, :] * info["std"][2] + info["mean"][2]
    return image

def forward_transform(image, dataset):
    info = transformations.get(dataset)
    image[:, 0, :, :] = (image[:, 0, :, :] - info["mean"][0]) / info["std"][0]
    image[:, 1, :, :] = (image[:, 1, :, :] - info["mean"][1]) / info["std"][1]
    image[:, 2, :, :] = (image[:, 2, :, :] - info["mean"][2]) / info["std"][2]
    return image


def compress_img(img, rate=75, format='jpeg', palette=256):
    from io import BytesIO

    byteImgIO = BytesIO()
    img.save(byteImgIO, format=format, quality=75)
    byteImgIO.seek(0)

    dataBytesIO = BytesIO(byteImgIO.read())
    compressed_img = Image.open(dataBytesIO)

    return compressed_img

def depth_img(img, depth=8):
    imageWithColorPalette = img.convert("P", palette=Image.ADAPTIVE, colors=depth)

    return imageWithColorPalette.convert("RGB")
