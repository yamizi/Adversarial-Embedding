import sys, os, math

sys.path.append("./")
from art.attacks import ProjectedGradientDescent
from metrics.attacks import get_attack_params, KerasClassifier
from experiments import logger, RANDOM_SEED
import numpy as np

from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, array_to_img, save_img, load_img
from art import NUMPY_DTYPE
from art.attacks import FastGradientMethod
from art.utils import get_labels_np_array, projection, random_sphere
from PIL import Image

def _load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data


class SpatialResilientPGD(ProjectedGradientDescent):

    nb_classes = 0    
    test_quality = 0

    def __init__(self, classifier, norm=np.inf, eps=.5, eps_step=0.1, max_iter=50, targeted=True, num_random_init=0,
                 batch_size=128):

        SpatialResilientPGD.rate_best = 0
        super(SpatialResilientPGD, self).__init__(classifier, norm, eps, eps_step, max_iter, targeted, num_random_init, batch_size)


    @staticmethod 
    def compress_batch(x, rate=75, format='jpeg', palette=256):
        from io import BytesIO
        
        if rate==100:
            return x
        X = []

        for i, _x in enumerate(x):
            img = array_to_img(_x)
            byteImgIO  = BytesIO()
            img.save(byteImgIO , format=format,quality=75)
            byteImgIO.seek(0)

            dataBytesIO = BytesIO(byteImgIO.read())
            compressed_img = Image.open(dataBytesIO)

            c_img = img_to_array(compressed_img)
            X.append(c_img/palette)

        return np.array(X)

    @staticmethod 
    def craft(model, x, target, epsilon=3., num_random_init=10, max_iter=100, quality=75,eps_step=0.1):
        
            
        target = to_categorical(target)
        norm = 2

        logger.info('Crafting spacially resilient PGD attack; norm:{} epsilon:{}'.format(norm, epsilon))

        attack_params, attack_name = get_attack_params("targetted_pgd", norm,epsilon)
        attack_params["eps_step"] = eps_step

        classifier = KerasClassifier(model=model)
        crafter = SpatialResilientPGD(classifier,num_random_init=num_random_init, max_iter=max_iter, eps=1.)
        crafter.set_params(**attack_params)

        adv_x = crafter.generate(x,target, quality)
        return adv_x


    def _compute(self, x, y, eps, eps_step, random_init, quality):
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:])
            adv_x = x.astype(NUMPY_DTYPE) + random_sphere(n, m, eps, self.norm).reshape(x.shape)

            if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
                clip_min, clip_max = self.classifier.clip_values
                adv_x = np.clip(adv_x, clip_min, clip_max)
        else:
            adv_x = x.astype(NUMPY_DTYPE)

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(adv_x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = adv_x[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            # Get compressed perturbation
            compressed_batch = SpatialResilientPGD.compress_batch(batch, rate=quality)
            perturbation = self._compute_perturbation(compressed_batch, batch_labels)

            # Apply perturbation and clip
            adv_x[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, eps_step)

        return adv_x


    def compute_success(self, classifier, x_clean, labels, x_adv, targeted=False, quality=75):
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
        #compressed_adv = SpatialResilientPGD.compress_batch(x_adv, rate=quality)
        compressed_adv = []
        #compressed_adv = x_adv
        adv_path= "./utils/adv.jpg"
        for adv in x_adv:
            quality = SpatialResilientPGD.test_quality if SpatialResilientPGD.test_quality else quality
            save_img(adv_path,adv, quality=quality)
            adv_x_post = _load_image(adv_path)
            compressed_adv.append(adv_x_post)

        compressed_adv = np.array(compressed_adv)
        
        adv_y = classifier.predict(compressed_adv)
        adv_preds = np.argmax(adv_y, axis=1)    
        rate = np.sum(adv_preds == np.argmax(labels, axis=1)) / x_adv.shape[0]

        return rate
    
    def generate(self, x, y, quality):
        self.targeted = True

        
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

        targets = y

        adv_x_best = None
        rate_best = 0.0

        for i_random_init in range(max(1, self.num_random_init)):
            adv_x = x.astype(NUMPY_DTYPE)

            for i_max_iter in range(self.max_iter):

                adv_x = self._compute(adv_x, targets, self.eps, self.eps_step,
                                      self.num_random_init > 0 and i_max_iter == 0, quality=quality)

                if self._project:
                    noise = projection(adv_x - x, self.eps, self.norm)
                    adv_x = x + noise

            rate = 100 * self.compute_success(self.classifier, x, targets, adv_x, self.targeted, quality=quality)
            logger.info('Success rate {}: {}'.format(i_random_init, rate))
            if rate > rate_best or adv_x_best is None:
                rate_best = rate
                adv_x_best = adv_x

        logger.info('Success rate of Spatially resilient PGD attack: %.2f%%', rate_best)

        SpatialResilientPGD.rate_best = rate_best
        return adv_x_best


def run_tests(use_gpu=True):
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    import tensorflow as tf
    from utils.adversarial_models import load_model

    model, x_train, x_test, y_train, y_test = load_model(
        dataset="cifar10", model_type="basic", epochs=25)
    
    quality = 100
    SpatialResilientPGD.test_quality = 75
    nb_elements = 100
    np.random.seed(RANDOM_SEED)
    y = np.random.randint(0,10,nb_elements)
    
    adv_x = SpatialResilientPGD.craft(model, x_test[:nb_elements],y, epsilon=3, max_iter=500, quality=quality, eps_step=0.01,num_random_init=50)
    adv_y = model.predict(adv_x)
    #logger.info("{}: {}".format(np.sort(-adv_y[0]).shape, list(zip(y, np.argmax(adv_y,axis=1)))))
 

    adv_path= "./utils/adv.jpg"
    save_img(adv_path,adv_x[0], quality=quality)

    
    adv_x_post = np.array([_load_image(adv_path)])
    adv_y_post = model.predict(adv_x_post)
    logger.info("{}-{}".format(np.argmax(adv_y_post,axis=1),np.argmax(adv_y,axis=1)))



if __name__ == "__main__":
    run_tests(False)
