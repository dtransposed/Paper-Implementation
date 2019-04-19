import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model, losses, optimizers, metrics
import numpy as np
import cv2



def preprocess_observation(observation):
    observation = tf.image.rgb_to_grayscale(observation)
    observation = tf.image.resize(observation, size=[110, 84])
    observation = observation[13:, :]
    observation = np.resize(observation,(84,84))
    observation = observation.astype('uint8')
    return observation

class MyModel(Model):
    def __init__(self, action_space):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(input_shape = (None,84,84,4), filters = 16, kernel_size = 8, strides = 4, activation='relu')
        self.conv2 = Conv2D(filters=32, kernel_size= 4, strides=2, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(action_space, activation='linear')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x






