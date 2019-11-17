
import numpy as np
from art.attacks import ProjectedGradientDescent, DeepFool, CarliniL2Method
from art.classifiers import KerasClassifier
from art import NUMPY_DTYPE
from art.utils import compute_success, get_labels_np_array, random_sphere, tanh_to_original, original_to_tanh
import keras
from utils.basic_cifar_cnn import get_model as basic_model, get_dataset as cifar_dataset
from utils.tiny_imagenet import tiny_imagenet_dataset
from utils.resnetCifar10 import run
from utils.feinman17 import get_model as get_feinman_model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import math

import os
import logging
logger = logging.getLogger(__name__)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size, scheduler=False, data_augmentation=False, data_folder=None):

    
    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)

    callbacks = []#, lr_reducer, lr_scheduler]

    if scheduler:
        callbacks = [lr_reducer, lr_scheduler]#[checkpoint, lr_reducer, lr_scheduler]

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'temp_model.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
       
    model_path = os.path.join(save_dir, model_name)
    mc = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True)
    callbacks.append(mc)


    if data_folder:
        datagen = ImageDataGenerator()
        train_it = datagen.flow_from_directory('{}/train/'.format(data_folder), class_mode='categorical')
        val_it = datagen.flow_from_directory('{}/validation/'.format(data_folder), class_mode='categorical')
        test_it = datagen.flow_from_directory('{}/test/'.format(data_folder), class_mode='categorical')

        history = model.fit_generator(train_it, steps_per_epoch=16, validation_data=val_it, validation_steps=8,epochs=epochs, verbose=2, workers=4,
                            callbacks=callbacks)

    # Run training, with or without data augmentation.
    elif not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True,
                callbacks=callbacks,
                verbose=2)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        steps_per_epoch= math.ceil(len(x_train) / batch_size)
        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=steps_per_epoch,
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=2, workers=4,
                            callbacks=callbacks)

    model = keras.models.load_model(model_path)
    os.remove(model_path)

    return model



class TrackedCW(CarliniL2Method):
    tracked_x = []

    def generate(self, x, y=None):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: If `self.targeted` is true, then `y_val` represents the target labels. Otherwise, the targets are
                the original class labels.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        x_adv = x.astype(NUMPY_DTYPE)
        if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
        else:
            clip_min, clip_max = np.amin(x), np.amax(x)

        # Assert that, if attack is targeted, y_val is provided:
        if self.targeted and y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.classifier.predict(x, logits=False))

        # Compute perturbation with implicit batching
        nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
        for batch_id in range(nb_batches):
            logger.debug('Processing batch %i out of %i', batch_id, nb_batches)

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x_adv[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]

            # The optimization is performed in tanh space to keep the adversarial images bounded in correct range
            x_batch_tanh = original_to_tanh(x_batch, clip_min, clip_max, self._tanh_smoother)

            # Initialize binary search:
            c = self.initial_const * np.ones(x_batch.shape[0])
            c_lower_bound = np.zeros(x_batch.shape[0])
            c_double = (np.ones(x_batch.shape[0]) > 0)

            # Initialize placeholders for best l2 distance and attack found so far
            best_l2dist = np.inf * np.ones(x_batch.shape[0])
            best_x_adv_batch = x_batch.copy()

            for bss in range(self.binary_search_steps):
                logger.debug('Binary search step %i out of %i (c_mean==%f)', bss, self.binary_search_steps, np.mean(c))
                nb_active = int(np.sum(c < self._c_upper_bound))
                logger.debug('Number of samples with c < _c_upper_bound: %i out of %i', nb_active, x_batch.shape[0])
                if nb_active == 0:
                    break
                lr = self.learning_rate * np.ones(x_batch.shape[0])

                # Initialize perturbation in tanh space:
                x_adv_batch = x_batch.copy()
                x_adv_batch_tanh = x_batch_tanh.copy()

                z, l2dist, loss = self._loss(x_batch, x_adv_batch, y_batch, c)
                attack_success = (loss - l2dist <= 0)
                overall_attack_success = attack_success

                for it in range(self.max_iter):
                    logger.debug('Iteration step %i out of %i', it, self.max_iter)
                    logger.debug('Average Loss: %f', np.mean(loss))
                    logger.debug('Average L2Dist: %f', np.mean(l2dist))
                    logger.debug('Average Margin Loss: %f', np.mean(loss-l2dist))
                    logger.debug('Current number of succeeded attacks: %i out of %i', int(np.sum(attack_success)),
                                 len(attack_success))

                    improved_adv = attack_success & (l2dist < best_l2dist)
                    logger.debug('Number of improved L2 distances: %i', int(np.sum(improved_adv)))
                    if np.sum(improved_adv) > 0:
                        best_l2dist[improved_adv] = l2dist[improved_adv]
                        best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]

                    active = (c < self._c_upper_bound) & (lr > 0)
                    nb_active = int(np.sum(active))
                    logger.debug('Number of samples with c < _c_upper_bound and lr > 0: %i out of %i',
                                 nb_active, x_batch.shape[0])
                    if nb_active == 0:
                        break

                    # compute gradient:
                    logger.debug('Compute loss gradient')
                    perturbation_tanh = -self._loss_gradient(z[active], y_batch[active], x_batch[active],
                                                             x_adv_batch[active], x_adv_batch_tanh[active],
                                                             c[active], clip_min, clip_max)

                    # perform line search to optimize perturbation
                    # first, halve the learning rate until perturbation actually decreases the loss:
                    prev_loss = loss.copy()
                    best_loss = loss.copy()
                    best_lr = np.zeros(x_batch.shape[0])
                    halving = np.zeros(x_batch.shape[0])

                    for h in range(self.max_halving):
                        logger.debug('Perform halving iteration %i out of %i', h, self.max_halving)
                        do_halving = (loss[active] >= prev_loss[active])
                        logger.debug('Halving to be performed on %i samples', int(np.sum(do_halving)))
                        if np.sum(do_halving) == 0:
                            break
                        active_and_do_halving = active.copy()
                        active_and_do_halving[active] = do_halving

                        lr_mult = lr[active_and_do_halving]
                        for _ in range(len(x.shape)-1):
                            lr_mult = lr_mult[:, np.newaxis]

                        new_x_adv_batch_tanh = x_adv_batch_tanh[active_and_do_halving] + \
                            lr_mult * perturbation_tanh[do_halving]
                        new_x_adv_batch = tanh_to_original(new_x_adv_batch_tanh, clip_min, clip_max,
                                                           self._tanh_smoother)
                        _, l2dist[active_and_do_halving], loss[active_and_do_halving] = self._loss(
                            x_batch[active_and_do_halving], new_x_adv_batch, y_batch[active_and_do_halving],
                            c[active_and_do_halving])

                        logger.debug('New Average Loss: %f', np.mean(loss))
                        logger.debug('New Average L2Dist: %f', np.mean(l2dist))
                        logger.debug('New Average Margin Loss: %f', np.mean(loss-l2dist))

                        best_lr[loss < best_loss] = lr[loss < best_loss]
                        best_loss[loss < best_loss] = loss[loss < best_loss]
                        lr[active_and_do_halving] /= 2
                        halving[active_and_do_halving] += 1
                    lr[active] *= 2

                    # if no halving was actually required, double the learning rate as long as this
                    # decreases the loss:
                    for d in range(self.max_doubling):
                        logger.debug('Perform doubling iteration %i out of %i', d, self.max_doubling)
                        do_doubling = (halving[active] == 1) & (loss[active] <= best_loss[active])
                        logger.debug('Doubling to be performed on %i samples', int(np.sum(do_doubling)))
                        if np.sum(do_doubling) == 0:
                            break
                        active_and_do_doubling = active.copy()
                        active_and_do_doubling[active] = do_doubling
                        lr[active_and_do_doubling] *= 2

                        lr_mult = lr[active_and_do_doubling]
                        for _ in range(len(x.shape)-1):
                            lr_mult = lr_mult[:, np.newaxis]

                        new_x_adv_batch_tanh = x_adv_batch_tanh[active_and_do_doubling] + \
                            lr_mult * perturbation_tanh[do_doubling]
                        new_x_adv_batch = tanh_to_original(new_x_adv_batch_tanh, clip_min, clip_max,
                                                           self._tanh_smoother)
                        _, l2dist[active_and_do_doubling], loss[active_and_do_doubling] = self._loss(
                            x_batch[active_and_do_doubling], new_x_adv_batch, y_batch[active_and_do_doubling],
                            c[active_and_do_doubling])
                        logger.debug('New Average Loss: %f', np.mean(loss))
                        logger.debug('New Average L2Dist: %f', np.mean(l2dist))
                        logger.debug('New Average Margin Loss: %f', np.mean(loss-l2dist))
                        best_lr[loss < best_loss] = lr[loss < best_loss]
                        best_loss[loss < best_loss] = loss[loss < best_loss]

                    lr[halving == 1] /= 2

                    update_adv = (best_lr[active] > 0)
                    logger.debug('Number of adversarial samples to be finally updated: %i', int(np.sum(update_adv)))

                    if np.sum(update_adv) > 0:
                        active_and_update_adv = active.copy()
                        active_and_update_adv[active] = update_adv
                        best_lr_mult = best_lr[active_and_update_adv]
                        for _ in range(len(x.shape) - 1):
                            best_lr_mult = best_lr_mult[:, np.newaxis]
                        x_adv_batch_tanh[active_and_update_adv] = x_adv_batch_tanh[active_and_update_adv] + \
                            best_lr_mult * perturbation_tanh[update_adv]
                        x_adv_batch[active_and_update_adv] = tanh_to_original(x_adv_batch_tanh[active_and_update_adv],
                                                                              clip_min, clip_max, self._tanh_smoother)
                        z[active_and_update_adv], l2dist[active_and_update_adv], loss[active_and_update_adv] = \
                            self._loss(x_batch[active_and_update_adv], x_adv_batch[active_and_update_adv],
                                       y_batch[active_and_update_adv], c[active_and_update_adv])
                        attack_success = (loss - l2dist <= 0)
                        overall_attack_success = overall_attack_success | attack_success

                # Update depending on attack success:
                improved_adv = attack_success & (l2dist < best_l2dist)
                logger.debug('Number of improved L2 distances: %i', int(np.sum(improved_adv)))

                if np.sum(improved_adv) > 0:
                    best_l2dist[improved_adv] = l2dist[improved_adv]
                    best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]

                c_double[overall_attack_success] = False
                c[overall_attack_success] = (c_lower_bound + c)[overall_attack_success] / 2

                c_old = c
                c[~overall_attack_success & c_double] *= 2
                c[~overall_attack_success & ~c_double] += (c - c_lower_bound)[~overall_attack_success & ~c_double] / 2
                c_lower_bound[~overall_attack_success] = c_old[~overall_attack_success]

            x_adv[batch_index_1:batch_index_2] = best_x_adv_batch
            rate = 100 * compute_success(self.classifier, x, y, x_adv, self.targeted)
            TrackedCW.tracked_x.append((x_adv,rate,batch_id, best_l2dist.mean()))

        logger.info('Success rate of C&W L_2 attack: %.2f%%',
                    100 * compute_success(self.classifier, x, y, x_adv, self.targeted))

        return x_adv


class TrackedDeepFool(DeepFool):
    tracked_x = []

class TrackedPGD(ProjectedGradientDescent):
    tracked_x = []

    def generate(self, x, y=None):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: The labels for the data `x`. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
                  Labels should be one-hot-encoded.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        
        from art import NUMPY_DTYPE
        from art.utils import compute_success, get_labels_np_array, projection

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError('Target labels `y` need to be provided for a targeted attack.')

            # Use model predictions as correct outputs
            targets = get_labels_np_array(self.classifier.predict(x))
        else:
            targets = y

        adv_x_best = None
        rate_best = 0.0

        for i_random_init in range(max(1, self.num_random_init)):
            adv_x = x.astype(NUMPY_DTYPE)
            noise = np.zeros_like(x)
            for i_max_iter in range(self.max_iter):

                adv_x = self._compute(adv_x, targets, self.eps, self.eps_step,
                                      self.num_random_init > 0 and i_max_iter == 0)
                if self._project:
                    noise = projection(adv_x - x, self.eps, self.norm)
                    adv_x = x + noise

                rate = 100 * compute_success(self.classifier, x, targets, adv_x, self.targeted)
                logger.info('Success rate of attack step: %.2f%%', rate)

                noise_norm  = 0
                if self.norm == np.inf:
                    noise_norm = np.sign(noise)
                elif self.norm == 1:
                    ind = tuple(range(1, len(noise.shape)))
                    noise_norm = np.sum(np.abs(noise), axis=ind, keepdims=True) 
                elif self.norm == 2:
                    ind = tuple(range(1, len(noise.shape)))
                    noise_norm = np.sqrt(np.sum(np.square(noise), axis=ind, keepdims=True))
                    
                TrackedPGD.tracked_x.append((adv_x,rate,i_max_iter, noise_norm))

            rate = 100 * compute_success(self.classifier, x, targets, adv_x, self.targeted)
            if rate > rate_best or adv_x_best is None:
                rate_best = rate
                adv_x_best = adv_x
            

        logger.info('Success rate of attack: %.2f%%', rate_best)

        return adv_x_best


def load_model(dataset="cifar10",model_type="basic",epochs=1, train_size=0, batch_size=64, data_augmentation=True):
    from keras.utils import multi_gpu_model
    from keras.utils import multi_gpu_utils
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    gpu_devices = [device for device in local_device_protos if device.device_type=="GPU"]
                    

    if model_type.find("h5") >-1:
        model_path = model_type
    else:
        model_name = "{}_{}_{}_{}_model.h5".format(dataset, model_type, epochs, data_augmentation)
        #model_name = "{}_{}_{}_model.h5".format(dataset, model_type, epochs)
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_path = os.path.join(save_dir, model_name)
        model = None

    if dataset=="cifar10":
        num_classes, x_train, y_train, x_test, y_test = cifar_dataset()
    elif dataset=="cifar100":
        num_classes, x_train, y_train, x_test, y_test = cifar_dataset("cifar100")
    elif dataset=="tiny_imagenet":
        num_classes, x_train, y_train, x_test, y_test = tiny_imagenet_dataset()
    # elif dataset=="mnist":
    #     num_classes, x_train, y_train, x_test, y_test = minst_dataset()

    
    if os.path.isfile(model_path):
        print("Loading existing model {}".format(model_path))
        try:
            model = keras.models.load_model(model_path)
            # if len(gpu_devices):
            #     model = multi_gpu_model(model, gpus=len(gpu_devices))
            if isinstance(model.layers[-2], keras.engine.training.Model):
                model = model.layers[-2]
                model.compile(optimizer="adam",
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])

                #print(model.summary())
        except Exception as e:
            print(e)
        
    else:
        print("Building new model {}".format(model_path))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        

        if model_type=="basic":
            model, _, _, _, _ = basic_model(epochs, batch_size, train_size=train_size, data_augmentation=data_augmentation,dataset=dataset)
        elif model_type=="mobilenet":
            model, _, _, _, _ = mobilenet(epochs, batch_size, train_size=train_size, data_augmentation=data_augmentation)
        elif model_type=="resnet":
            model, _, _, _, _ = manual_resnet(epochs, batch_size, train_size=train_size, data_augmentation=data_augmentation,dataset=dataset)

        elif model_type[:7]=="feinman":
            model = get_feinman_model(dataset)

        model = train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size, True, data_augmentation)

        # Score trained model.
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        # Save model and weights
        model.save(model_path)
        print('Saved trained model at {} '.format(model_path))

    return model, x_train, x_test, y_train, y_test


def sadl_mode():
    layers = [
            keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)),
            keras.layers.Activation("relu"),
            keras.layers.Conv2D(32, (3, 3), padding="same"),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, (3, 3), padding="same"),
            keras.layers.Activation("relu"),
            keras.layers.Conv2D(64, (3, 3), padding="same"),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(128, (3, 3), padding="same"),
            keras.layers.Activation("relu"),
            keras.layers.Conv2D(128, (3, 3), padding="same"),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1024, kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Activation("relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Activation("relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10),
        ]
        


def manual_resnet(epochs=1, batch_size=128, train_size=0,data_augmentation=False, dataset="cifar10"):

    if (dataset=="cifar10" or dataset=="mnist"):
        num_classes = 10
    else:
        num_classes = 100
    return run(scheduler=True, adam=True, epochs=epochs, batch_size=batch_size, train_size=train_size, data_augmentation=data_augmentation, num_classes=num_classes) 


def mobilenet(epochs=1, batch_size=128, train_size=0, data_augmentation=False):
    from keras.applications.mobilenet import MobileNet
    from keras.layers import Dense, Input, Dropout
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.datasets import cifar10

    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if train_size:
        x_train = x_train[:train_size]
        y_train = y_train[:train_size]

    input_tensor = Input(shape=input_shape)
    base_model = MobileNet(
        include_top=False,
        weights=None,#'imagenet',
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling='avg')

    for layer in base_model.layers:
        layer.trainable = True 
        
    op = keras.layers.Dense(256, activation='relu')(base_model.output)
    op = keras.layers.Dropout(.25)(op)
    output_tensor = keras.layers.Dense(num_classes, activation='softmax')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

    return model, x_train, x_test, y_train, y_test

if __name__ == "__main__":
    classifier, x_train, x_test, y_train, y_test = mobilenet()
    keras_model = KerasClassifier(model=classifier)
    epsilon = .1 # Maximum perturbation
    adv_crafter = TrackedPGD(keras_model)
    x_test_adv = adv_crafter.generate(x=x_test)
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy on adversarial sample: %.2f%%" % (acc * 100))

    last =  adv_crafter.tracked_x[0]
    print("nb artificial adversarial {} non adversarial {}:{}".format(len(x_test_adv), len(last[0]), last[1]))