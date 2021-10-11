from comet_ml import Experiment
from utils import init_comet
from exploration.torch_layers import run as run_attack
import torch

if __name__ == '__main__':
    parameters = {"use_hidden": False, "pretrained": True, "criterion": "mse", "algorithm": "mifgsm", "max_eps": 16/255,
                  "norm": "Linf", "max_iter": 1000, "eps_step": 0.05, "num_random_init": 5, "batch_size": 100,
                  "success_threshold": 0.9, "bpp": 0.01, "lib": "torchattack", "model": "cifar10_resnet20",
                  "dataset": "div2k", "strategy": "binary", "random_noise": "fading","binary_message":None}

    name = "steganography_transfer"
    experiment = init_comet(args=parameters, project_name=name)

    reload = True

    if reload:
        _, success1, x_test_adv, x_test = run_attack(parameters, name=name, experiment=experiment)
        torch.save(x_test, "x_test.pt")
        torch.save(x_test_adv, "x_test_adv.pt")

    else:
        x_test = torch.load("x_test.pt")
        x_test_adv = torch.load("x_test_adv.pt")

    from robustbench.utils import load_model

    models = ["Gowal2020Uncovering_70_16_extra","Gowal2020Uncovering_28_10_extra","Wu2020Adversarial_extra","Carmon2019Unlabeled","Sehwag2021Proxy"]

    for model_name in models:
        print("loading ", model_name)
        model = load_model(model_name=model_name, norm='Linf')
        model = model.cuda()

        with torch.no_grad():
            out1 = model(x_test.cuda())
            out2 = model(x_test_adv.cuda())

            classes1 = torch.argmax(out1, 1)
            classes2 = torch.argmax(out2, 1)

            robust_accuracy = (classes1 == classes2).float().mean()
            print(model_name, robust_accuracy)
            experiment.log_metric("robust_acc_{}".format(model_name),robust_accuracy)

        del model
        torch.cuda.empty_cache()
    equal = out1==out2
