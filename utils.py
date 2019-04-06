import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model, losses, optimizers, metrics
import numpy as np


def preprocess_observation(observation):
    observation = tf.image.rgb_to_grayscale(observation)
    observation = tf.image.resize(observation, size=[110, 84])
    observation = observation[26:110, :]
    observation = observation /255
    observation = np.resize(observation,(84,84))
    return observation

class MyModel(Model):
    def __init__(self, action_space):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(input_shape = (None,4,84,84), filters = 16, kernel_size = 8, strides = 4, activation='relu')
        self.conv2 = Conv2D(filters=32, kernel_size= 4, strides=2, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(action_space, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        value = tf.math.reduce_max(x)
        action = tf.math.argmax(x,1)
        return action, value

def show_image(numpy_array):
    plt.imshow(numpy_array)
    plt.show()
    plt.close()

def show_image_sequence(numpy_array):

    for i in range(4):
        numpy_array1 = numpy_array[i,:,:,:]
        numpy_array2 = numpy_array1[:,:,0]
        plt.imshow(numpy_array2)
        plt.show()
        plt.close()




