import sys, os, math

sys.path.append("./")
from art.classifiers import KerasClassifier
from art.attacks import ProjectedGradientDescent
from metrics.attacks import get_attack_params
from experiments import logger, RANDOM_SEED
import numpy as np

from art.attacks import FastGradientMethod
from art.utils import get_labels_np_array, projection, random_sphere
from keras.utils import to_categorical

import random

class SATA(ProjectedGradientDescent):

    nb_classes_per_img = 0
    power = 2
    num_cover_init=1000
    success_rates = []
    last_split = []

    def __init__(self, classifier, norm=np.inf, eps=.5, eps_step=0.1, max_iter=50, targeted=True, num_random_init=0,
                 batch_size=128):

        SATA.rate_best = 0
        super(SATA, self).__init__(classifier, norm, eps, eps_step, max_iter, targeted, num_random_init, batch_size)

    
    @staticmethod
    def _split_msg(msg, chunk_size, nb_classes_per_img):
    
        chunks = len(msg)
        groups = [[]]
        last_group = []
        for i, integer in enumerate(msg):
            if len(last_group)==chunk_size or integer in last_group:
                groups[-1].append(int("".join(last_group)))
                last_group = []

                if len(groups[-1]) >=nb_classes_per_img:
                    groups.append([])

            last_group.append(integer)

        groups[-1].append(int("".join(last_group)))    

        # groups elements are images, groups sub_elements are the classes to encode at every image
        return groups

    @staticmethod
    def embed_message(model, x, msg,epsilon=1., num_random_init=10, max_iter=100, class_density=0.7, eps_step=0.1, groups_only=False,num_classes=0,nb_classes_per_img=0):

        if num_classes==0:
            num_classes = model.output_shape[1]
        chunk_size = int(math.log(num_classes)/math.log(10))
        
        if nb_classes_per_img==0:
            nb_classes_per_img = int(class_density*num_classes)
        
        norm = 2
        #SATA.power = 2.5
        SATA.success_rates = []

        groups = SATA._split_msg(msg, chunk_size, nb_classes_per_img)
        SATA.last_split = groups

        if groups_only:
            return groups

        grps_len = [len(grp) for grp in groups]
        threshold = 1/ (max(grps_len)+1)

        attack_params, attack_name = get_attack_params("targetted_pgd", norm,epsilon)
        #print("####",model)

        import keras
        # from keras.models import Sequential
        # from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

        # model = Sequential()
        # model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(28, 28, 1)))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(23, 23, 4)))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Flatten())
        # model.add(Dense(100, activation='relu'))
        # model.add(Dense(10, activation='softmax'))


        #model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
        #print("embed attack shape",x.shape,np.array(groups).shape)

        classifier = KerasClassifier(model=model, use_logits=False)
        crafter = SATA(classifier,num_random_init=num_random_init, max_iter=max_iter, eps=1., eps_step=eps_step)
        crafter.nb_classes_per_img = nb_classes_per_img
        crafter.set_params(**attack_params)
        
        adv_x, ref_x = crafter.generate(x,groups, threshold, nb_classes=num_classes)

        return adv_x, ref_x, SATA.rate_best

    @staticmethod 
    def craft(model, x, order, epsilon=1., max_iter=100):
        nb_elements = x.shape[0]
        if len(order.shape) == 1:
            nb_classes= len(order)
            order = np.repeat(np.array([order]),nb_elements, axis=0)
        else:
            nb_classes= order.shape[1]
            
        threshold = 1/ (nb_classes+1)
        norm = 2

        logger.info('Crafting Sorted PGD attack; norm:{} threshold:{} epsilon:{}'.format(norm, (threshold, SATA.power), epsilon))

        attack_params, attack_name = get_attack_params("targetted_pgd", norm,epsilon)
        classifier = KerasClassifier(model=model)
        crafter = SATA(classifier,num_random_init=num_random_init, max_iter=max_iter, eps=1.)
        crafter.nb_classes = nb_classes
        crafter.set_params(**attack_params)

        adv_x,ref_x = crafter.generate(x,order, threshold)
        return adv_x


    def _compute(self, x, y, eps, eps_step, random_init):
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:])
            #adv_x = x.astype(NUMPY_DTYPE) + random_sphere(n, m, eps, self.norm).reshape(x.shape)
            adv_x = x+ random_sphere(n, m, eps, self.norm).reshape(x.shape)
            if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
                clip_min, clip_max = self.classifier.clip_values
                adv_x = np.clip(adv_x, clip_min, clip_max)
        else:
            adv_x = x#.astype(NUMPY_DTYPE)

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(adv_x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = adv_x[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation(batch, batch_labels)

            # Apply perturbation and clip
            adv_x[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, eps_step)

        return adv_x

    def _compute_perturbation(self, batch, batch_labels):
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.classifier.loss_gradient(batch, batch_labels) * (1 - 2 * int(self.targeted))

        # Apply norm bound
        if self.norm == np.inf:
            grad = np.sign(grad)
        elif self.norm == 1:
            ind = tuple(range(1, len(batch.shape)))
            grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
        elif self.norm == 2:
            ind = tuple(range(1, len(batch.shape)))
            grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
        assert batch.shape == grad.shape

        return grad

    def compute_success(self, classifier, x_clean, labels, x_adv, targeted=False):
        """
        Compute the success rate of an attack based on clean samples, adversarial samples and targets or correct labels.

        :param classifier: Classifier used for prediction.
        :type classifier: :class:`.Classifier`
        :param x_clean: Original clean samples.
        :type x_clean: `np.ndarray`
        :param labels: Correct labels of `x_clean` if the attack is untargeted, or target labels of the attack otherwise.
        :type labels: `np.ndarray`
        :param x_adv: Adversarial samples to be evaluated.
        :type x_adv: `np.ndarray`
        :param targeted: `True` if the attack is targeted. In that case, `labels` are treated as target classes instead of
            correct labels of the clean samples.s
        :type targeted: `bool`
        :return: Percentage of successful adversarial samples.
        :rtype: `float`
        """
        adv_y = classifier.predict(x_adv)

        adv_sort = np.argsort(-adv_y, axis=1)[:, 0:self.nb_classes_per_img]#.reshape(-1)
        real_sort = np.argsort(-labels, axis=1)[:, 0:self.nb_classes_per_img]#.reshape(-1)

        print(adv_sort.shape,real_sort.shape)
        equal = adv_sort == real_sort
        equal = [all(e) for e in equal]
        rate = np.sum(np.array(equal)) / real_sort.shape[0]

        return rate, equal
    
    def generate(self, x_all, order, threshold=0.1, nb_classes = 10):
        self.targeted = True

        y0 = []
        
        for o in order:
            y = [0] * nb_classes
            for i, j in enumerate(o):
                y[j] = max(0, math.pow(1-i*threshold,SATA.power))
            y0.append(y)
            #print(y)
        
        y = np.array(y0)

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

        targets = y.copy()
        logger.info('Nb images to embed the message {}'.format(targets.shape[0]))

        adv_x_best_index = [False]*len(order)
        adv_x_best = None
        rate_best = 0.0
        

        for i in range(SATA.num_cover_init):
            logger.info('Random inputs pick iteration index {}'.format(i))
            random.shuffle(x_all)
            x = x_all.copy()[0:len(order)]
            SATA.success_rates.append([])
            for i_random_init in range(max(1, self.num_random_init)):
                adv_x = x#.astype(NUMPY_DTYPE)
                SATA.success_rates[-1].append([])
                
                for i_max_iter in range(self.max_iter):

                    adv_x = self._compute(adv_x, targets, self.eps, self.eps_step,
                                        self.num_random_init > 0 and i_max_iter == 0)

                    if self._project:
                        noise = projection(adv_x - x, self.eps, self.norm)
                        adv_x = x + noise

                adv_x = np.array([adv_x[i] if adv_x_best is None or not adv_x_best_index[i] else adv_x_best[i] for i in range(len(adv_x))])
                rate, adv_x_best_index = self.compute_success(self.classifier, x, targets, adv_x, self.targeted)
                rate = 100 * rate
                SATA.success_rates[-1][-1].append(rate)

                if adv_x_best is None:
                    adv_x_best = adv_x
                    rate_best = rate

                elif rate > rate_best:
                    rate_best = rate
                    adv_x_best = np.where(adv_x_best_index==True,adv_x,adv_x_best)

                logger.info('Success rate of SATA attack: %.2f%%', rate_best)
                SATA.rate_best = rate_best
                if rate_best ==100:
                    return adv_x_best, x

        return adv_x_best,x


def run_tests(use_gpu=False):
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    import tensorflow as tf
    from utils.adversarial_models import load_model

    model, x_train, x_test, y_train, y_test = load_model(
        dataset="cifar10", model_type="basic", epochs=25)
    

    
    np.random.seed(RANDOM_SEED)
    

    nb_elements = 10
    nb_classes = 3
    order = np.array([2,3,5,8,9,4,6])
    order = order[0:nb_classes]

    order = np.random.randint(0,10,(nb_elements,nb_classes))

    SATA.power = 1
    adv_x = SATA.craft(model, x_test[:nb_elements],order, epsilon=3., max_iter=200)
    adv_y = model.predict(adv_x)
    logger.info("{}: {}".format(np.sort(-adv_y[0]).shape, list(zip(order, np.argsort(-1*adv_y,axis=1)))))


    



if __name__ == "__main__":
    run_tests()
