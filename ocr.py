# Filepaths, Numpy, Tensorflow
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

from extra_keras_datasets import emnist
#(input_train, target_train), (input_test, target_test) = emnist.load_data(type='bymerge')

def train():
    (x_train, y_train), (x_test, y_test) = emnist.load_data(type='byclass')

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    fixed_x_train = [[[]]]
    fixed_y_train = []
    
    for i in range(x_train.shape[0]):
        if y_train[i] < 36:
            fixed_x_train = fixed_x_train + [x_train[i][:][:]]
            fixed_y_train = fixed_y_train + [y_train[i]]

    fixed_x_test = [[[]]]
    fixed_y_test = []
    for i in range(x_test.shape[0]):
        if y_test[i] < 36:
            fixed_x_test = fixed_x_test + [x_test[i][:][:]]
            fixed_y_test = fixed_y_test + [y_test[i]]
    
    fixed_x_train = fixed_x_train[1:][:][:]
    fixed_x_test = fixed_x_test[1:][:][:]

    fixed_x_train = tf.convert_to_tensor(fixed_x_train)
    fixed_y_train = tf.convert_to_tensor(fixed_y_train)

    fixed_x_test = tf.convert_to_tensor(fixed_x_test)
    fixed_y_test = tf.convert_to_tensor(fixed_y_test)

    model = tf.keras.models.Sequential([
    Flatten(),
    Dense(1152, activation="relu"),
    Dropout(0.36),
    Dense(576, activation="relu"),
    Dropout(0.36),
    Dense(288, activation="relu"),
    Dropout(0.36),
    Dense(144, activation="relu"),
    Dropout(0.36),
    Dense(72,activation="relu"), 
    Dropout(0.36),
    Dense(36,activation="softmax")
    ])

    predictions = model(fixed_x_train).numpy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
    model.fit(fixed_x_train, fixed_y_train, epochs=25)
    # loss, acc = model.evaluate(fixed_x_test, fixed_y_test, verbose=2)
    # print("Trained model, accuracy: {:5.2f}%".format(100 * acc))

    model.save("ocr_model", save_format="h5")

    model.evaluate(fixed_x_test,  fixed_y_test, verbose=2)
    model.summary()

def predict(images):
    model = tf.keras.models.load_model("ocr_model")
    # return convert_result(model.predict(images))
    # print(model.predict(images).shape)
    return convert_result(np.argmax(model.predict(images)))

def convert_result(num):
    real = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", \
        "X", "Y", "Z"]
    return real[num]
