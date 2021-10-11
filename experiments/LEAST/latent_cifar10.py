import logging
logging.basicConfig(level=logging.DEBUG)

from utils.adversarial_models import load_model, load_dataset
from art.classifiers import KerasClassifier
from art.attacks import ProjectedGradientDescent
from sklearn.metrics import mean_squared_error,mean_absolute_error
from matplotlib import pyplot as plt

import numpy as np
import keras
from art.utils import compute_success


model, x_train, x_test, y_train, y_test = load_model(
    dataset="cifar10", model_type="basic", epochs=25)



#model2 = keras.Model(model.input, Flatten()(model.layers[-2].output))

model2 = keras.Model(model.input, keras.layers.Activation("softmax",name="model2_activation_1")(model.layers[-4].output))
model2.compile(optimizer="adam", loss=keras.losses.kullback_leibler_divergence, metrics=["mse"]) #"categorical_crossentropy" keras.losses.mean_squared_error
y_train2 = np.argmax(model2.predict(x_train),axis=1)

classifier = KerasClassifier(model=model2, use_logits=False)
atk = ProjectedGradientDescent(classifier, num_random_init=5,max_iter=500,eps_step=0.01, eps=0.05, targeted=True)


inputs = x_test #x_train[:5]
_, _ ,_, adv_inputs, _ = load_dataset("cifar10")
np.random.shuffle(adv_inputs)

nb_elements = 100
inputs = inputs[:nb_elements]
adv_inputs = adv_inputs[:nb_elements]

out1 = model2.predict(inputs)

#adv1 = atk.generate(x_train[:5])
#out_adv1 = model2.predict(adv1)
#success = compute_success(classifier, x_train[:5],y_train2[:5],adv1)
#print(success)

#plt.imshow(x_train[0].squeeze(), cmap='gray')
#plt.show()
#plt.imshow(adv1[0].squeeze(), cmap='gray')
#plt.show()

out2 = model2.predict(adv_inputs)
adv2 = atk.generate(inputs,out2)
out_adv2 = model2.predict(adv2)

success = compute_success(classifier, inputs,out1,adv2)
success_targeted = compute_success(classifier, inputs,out2,adv2, targeted=True)

print(success,success_targeted)


good_targeted = np.argmax(out_adv2, axis=1) == np.argmax(out2, axis=1)
good_adv, good_clean = adv2[good_targeted], inputs[good_targeted]

plt.imshow(good_adv[0].squeeze(), cmap='gray')
plt.show()
plt.imshow(good_clean[0].squeeze(), cmap='gray')
plt.show()

print(out1.shape)
