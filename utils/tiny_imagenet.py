import os
dataset_path = "../datasets/tiny-imagenet-200"

def tiny_imagenet_dataset(train_size=0, test_size=0, resize=None):
    num_classes = 200
    #, x_train, y_train, x_test, y_test = 

    train_path = "{}/{}".format(dataset_path,"train")
    classes = [i for i in os.listdir(train_path) if os.path.isdir(i)] 