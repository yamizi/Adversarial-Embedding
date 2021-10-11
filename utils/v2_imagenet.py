import math
import tensorflow_datasets as tfds
import tensorflow as tf
from utils.resnetCifar10 import  run
from utils.basic_cifar_cnn import get_model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical as to_cat

def normalize_img(image):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255.

def v2():

    (ds_train2, ds_test2), ds_info2 = tfds.load(
        'imagenet_resized/32x32',
        split=['train', 'validation'],
        shuffle_files=True,
        with_info=True,
        as_supervised=False
    )

    ds_train = ds_train2.map(normalize_img)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info2.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test2#.map(normalize_img)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


    np_train = tfds.as_numpy(ds_train)
    np_test = tfds.as_numpy(ds_test)

    for images, labels in ds_train.take(1):  # only take first element of dataset
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        print(numpy_labels)


    x_train, y_train = np_train["image"], np_train["label"] # seperate the x and y
    x_test, y_test = np_test["image"], np_test["label"]

    run(scheduler=True, adam=True, dataset=((x_train, (x_test, y_test)), np_test))

#v2()


(ds_train, ds_test), ds_info = tfds.load(
    'imagenet_resized/32x32',
    split=['train','validation'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

batch_size = 128
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

np_train = tfds.as_numpy(ds_train)

nb_classes = 1000
x_train = [

]

y_train = []
x_test = [

]

y_test = []
max_train = 10000
max_test = 500
epochs = 50
callbacks = []

for i, example in enumerate(np_train):
  image, label = example[0], example[1]
  x_train.append(image)
  y_train.append(label)

  if i*batch_size >=max_train:
      break

for i, example in enumerate(np_train):
  image, label = example[0], example[1]
  x_test.append(image)
  y_test.append(label)

  if i*batch_size >=max_test:
      break

x_train, x_test = np.concatenate(x_train), np.concatenate(x_test)
y_train, y_test = np.concatenate(y_train), np.concatenate(y_test)
y_train_cat, y_test_cat = to_cat(y_train, nb_classes), to_cat(y_test, nb_classes)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#x_train_mean = np.mean(x_train, axis=0)
#x_train -= x_train_mean
#x_test -= x_train_mean

dataset = (nb_classes,x_train,y_train,x_test,y_test)
input_shape=(32,32,3)
input_shape=(224,224,3)

model = tf.keras.applications.MobileNetV2(
    input_shape=input_shape,
    include_top=True,
    classes=nb_classes,
    weights = "imagenet"
)
#model.compile(loss='categorical_crossentropy',
#  optimizer="adam",
#  metrics=['accuracy'])
import scipy
new_shape = (224,224)
x_resize = np.array([scipy.misc.imresize(X, new_shape) for X in x_test])
y_test_predict = model.predict(x_resize)
y_test_predict_class = np.argmax(y_test_predict, axis=1)
print(y_test_predict_class == y_test)
"""
datagen = ImageDataGenerator(
target_size=(224,224),
    batch_size=256,
    classes=None)
datagen.fit(x_train)
steps_per_epoch = math.ceil(len(x_train) / batch_size)
model.fit_generator(datagen, steps_per_epoch=steps_per_epoch, epochs=100)

from keras.callbacks import  TensorBoard
tensorboard_callback = TensorBoard(log_dir="./tensorboard/mobilenet", histogram_freq=0)
callbacks.append(tensorboard_callback)

model.fit(x_train, y_train_cat,
  batch_size=batch_size,
  epochs=epochs,
  validation_data=(x_test, y_test_cat),
  shuffle=True,
  callbacks=callbacks)
"""

#num_classes = 10, n = 3, version = 1, adam=False, scheduler=False, train_size=0, dataset=None
#run(epochs=50, dataset = dataset, n=3, version=2, adam=True)
#model = get_model(100,dataset=dataset, data_augmentation=True)
