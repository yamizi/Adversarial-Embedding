from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten, Conv2D, Dropout,MaxPooling2D 
from keras.regularizers import l2

from keras.optimizers import Adam

"""
From https://github.com/rfeinman/detecting-adversarial-samples/blob/master/detect/util.py
"""

def get_model(dataset='mnist'):
    """
    Takes in a parameter indicating which model type to use ('mnist',
    'cifar' or 'svhn') and returns the appropriate Keras model.
    :param dataset: A string indicating which dataset we are building
                    a model for.
    :return: The model; a Keras 'Sequential' instance.
    """
    assert dataset in ['mnist', 'cifar', 'cifar10','svhn'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    if dataset == 'mnist':
        # MNIST model
        layers = [
            Conv2D(64, (3, 3), padding='valid', input_shape=(28, 28, 1)),
            Activation('relu'),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(128),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')
        ]
    elif (dataset == 'cifar10' or dataset == 'cifar'):
        # CIFAR-10 model
        layers = [
            Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
            Activation('relu'),
            Conv2D(32, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), padding='same'),
            Activation('relu'),
            Conv2D(64, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), padding='same'),
            Activation('relu'),
            Conv2D(128, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation('relu'),
            Dropout(0.5),
            Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')
        ]
    else:
        # SVHN model
        layers = [
            Conv2D(64, (3, 3), padding='valid', input_shape=(32, 32, 3)),
            Activation('relu'),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(512),
            Activation('relu'),
            Dropout(0.5),
            Dense(128),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')
        ]

    model = Sequential()
    for layer in layers:
        model.add(layer)

    model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
    return model