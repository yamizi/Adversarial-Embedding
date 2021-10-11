from attacks import prepare_message, Identity
from utils.loader import DataLoader
from attacks.dag import DAG
from torchvision import models, transforms
import numpy as np
import random, string
import torch

def run(parameters):
    transform = transforms.Compose([            #[1]
     transforms.Resize(256),                    #[2]
     transforms.CenterCrop(224),                #[3]
     transforms.ToTensor(),                     #[4]
     transforms.Normalize(                      #[5]
     mean=[0.485, 0.456, 0.406],                #[6]
     std=[0.229, 0.224, 0.225]                  #[7]
     )])

    validation = DataLoader('../SteganoGAN/research/data/div2k/val/', limit=np.inf, shuffle=True, batch_size=parameters.get("batch_size"), transform=transform)
    batch_t = None
    for batch in validation:
        batch_t = batch[0]
        break

    model = models.resnet18(pretrained=parameters.get("pretrained"))

    if parameters.get("use_hidden"):
        model.fc = Identity()
        model.avgpool = Identity(activation=True)

    model.eval()
    out = model(batch_t)


    min_val,max_val = out.min().item(),out.max().item()

    cover = batch_t[0]
    fix =  7
    capacity = out.shape[1]
    letters = string.ascii_letters
    message = ''.join(random.choice(letters) for i in range(int(cover.shape[1] * cover.shape[2] / fix * parameters.get("bpp",1))))
    encoded = prepare_message(message, capacity=capacity, pixels = cover.shape[2] * cover.shape[1],min_val=min_val,max_val=max_val)


    x_test = batch_t.float()
    target_labels = torch.from_numpy(encoded).unsqueeze(0).type(torch.float).repeat(x_test.shape[0], 1)
    dag = DAG(model,batch_t, out,target_labels, threshold=(min_val+max_val)/2)
    print(dag.shape)


if __name__ == '__main__':

    #experiment = init_comet()

    parameters = {"use_hidden" : False, "pretrained" : True, "criterion" : "mse", "algorithm":"pgd", "max_eps":5,
                  "norm":"Linf", "max_iter":1000, "eps_step":0.05,"num_random_init":5, "batch_size":4 ,
                  "success_threshold": 0.9, "bpp":0.015}

    run(parameters)
