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
    return convert_result(model.predict(images))

def convert_result(num):
    real = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W" \
        "X", "Y", "Z"])
    return real[num]

#train()



# def main():
#     model = VGGModel()

#     # checkpoint_path = "checkpoints" + os.sep + \
#     #         "vgg_model" + os.sep + timestamp + os.sep
#     #     logs_path = "logs" + os.sep + "vgg_model" + \
#     #         os.sep + timestamp + os.sep
    
#     # Print summaries for both parts of the model
#         model.vgg16.summary()
#         model.head.summary()

#         # Load base of VGG model
#         model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

# def train(model, datasets, checkpoint_path, logs_path, init_epoch):
#     """ Training routine. """

#     # Keras callbacks for training
#     callback_list = [
#         tf.keras.callbacks.TensorBoard(
#             log_dir=logs_path,
#             update_freq='batch',
#             profile_batch=0),
#         ImageLabelingLogger(logs_path, datasets),
#         CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
#     ]

#     # Begin training
#     model.fit(
#         x=datasets.train_data,
#         validation_data=datasets.test_data,
#         epochs=hp.num_epochs,
#         batch_size=None,
#         callbacks=callback_list,
#         initial_epoch=init_epoch,
#     )


# class VGGModel(tf.keras.Model):
#     def __init__(self):
#         super(VGGModel, self).__init__()

#         self.optimizer = tf.keras.optimizers.Adam(hp.learning_rate)

#         # Don't change the below:
        
#         self.vgg16 = [
#             # Block 1
#             Conv2D(64, 3, 1, padding="same",
#                    activation="relu", name="block1_conv1"),
#             Conv2D(64, 3, 1, padding="same",
#                    activation="relu", name="block1_conv2"),
#             MaxPool2D(2, name="block1_pool"),
#             # Block 2
#             Conv2D(128, 3, 1, padding="same",
#                    activation="relu", name="block2_conv1"),
#             Conv2D(128, 3, 1, padding="same",
#                    activation="relu", name="block2_conv2"),
#             MaxPool2D(2, name="block2_pool"),
#             # Block 3
#             Conv2D(256, 3, 1, padding="same",
#                    activation="relu", name="block3_conv1"),
#             Conv2D(256, 3, 1, padding="same",
#                    activation="relu", name="block3_conv2"),
#             Conv2D(256, 3, 1, padding="same",
#                    activation="relu", name="block3_conv3"),
#             MaxPool2D(2, name="block3_pool"),
#             # Block 4
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block4_conv1"),
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block4_conv2"),
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block4_conv3"),
#             MaxPool2D(2, name="block4_pool"),
#             # Block 5
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block5_conv1"),
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block5_conv2"),
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block5_conv3"),
#             MaxPool2D(2, name="block5_pool")
#         ]

#         # TODO: Make all layers in self.vgg16 non-trainable. This will freeze the
#         #       pretrained VGG16 weights into place so that only the classificaiton
#         #       head is trained.
#         for layer in self.vgg16:
#                layer.trainable = False

#         self.head = [
#                Flatten(),
#                Dropout(0.28),
#                Dense(hp.num_classes,activation="softmax")
#         ]

#         # Don't change the below:
#         self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
#         self.head = tf.keras.Sequential(self.head, name="vgg_head")

#     def call(self, x):
#         """ Passes the image through the network. """

#         x = self.vgg16(x)
#         x = self.head(x)

#         return x

#     @staticmethod
#     def loss_fn(labels, predictions):
#         """ Loss function for model. """

#         # TODO: Select a loss function for your network (see the documentation
#         #       for tf.keras.losses)

#         return tf.keras.losses.sparse_categorical_crossentropy(labels,predictions)
