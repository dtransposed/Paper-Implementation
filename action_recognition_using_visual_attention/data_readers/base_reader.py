import tensorflow as tf
import keras


class BaseReader(object):
    def __init__(self, dataset_directory, batch_size):
        self.dataset_dir = dataset_directory

        self.X_train = None
        self.y_train = None

        self.X_val = None
        self.y_val = None

        self.X_test = None
        self.y_test = None

        self.train_ds = None
        self.test_ds = None
        self.val_ds = None

        self.batch_size = batch_size

    def load_image_vgg(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        return img

    def preprocess_image_vgg(self, img):
        img = keras.applications.vgg16.preprocess_input(img)
        return img

