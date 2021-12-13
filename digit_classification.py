import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Reshape, LeakyReLU, BatchNormalization
from tensorflow.math import log

## generator
class DigitClassifier(tf.keras.Model):
    def __init__(self):
        super(DigitClassifier, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.text_embedding = Sequential([
            Dense(128, activation=None), 
            LeakyReLU(alpha=0.05)])
        ## explore leaky relu activations

        self.deconv = Sequential([
            Dense(8*8*256),
            Reshape((8, 8, 256)),
            Conv2DTranspose(128, [5,5], strides=(1, 1), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2DTranspose(64, [5,5], strides=(2, 2), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2DTranspose(32, [5,5], strides=(2, 2), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2DTranspose(3, [5,5], strides=(2, 2), padding='same', activation='tanh'),
            ])

    def call(self, latent_rep, text):
        """
        param latent_rep: latent space representation to be turned into image
        param text      : text embedding to concat with image
        returns: fake generated image
        """
        embedded_text = self.text_embedding(text)
        x = tf.concat([latent_rep, embedded_text], axis=-1)
        fimg = self.deconv(x)
        return fimg

    def loss(fake_output):
        ## TODO: Verify that this matches what the paper is looking for 
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return bce(tf.ones_like(fake_output), fake_output)
