from .base_reader import BaseReader
import tensorflow as tf
import numpy as np
import os
import glob
import cv2


class HMDBDataReader(BaseReader):

    def __init__(self, dataset_directory, batch_size, sequence_len, base_type):
        super().__init__(dataset_directory, batch_size)

        self.sequence_len = sequence_len
        self.base_type = base_type

        if self.base_type == 'VGG':
            self.file_ending = "_vgg.npy"

        else:
            raise ValueError("Improper base type!")

    def load_np_cube(self, cube_path_name):
        cube = np.load(cube_path_name.decode('utf-8'))
        return cube,

    def save_4d_to_numpy(self):
        print('Saving video sequences as numpy arrays...')
        folders = ['train', 'test', 'val']
        for folder in folders:
            instances_list = os.listdir(os.path.join(self.dataset_dir, folder))
            for instance in instances_list:
                if glob.glob(os.path.join(self.dataset_dir, folder,instance, '*.npy')):
                    pass
                else:
                    frame_list = [x for x in os.listdir(os.path.join(self.dataset_dir, folder, instance))
                                  if x.endswith('.jpg')]
                    frame_list.sort()
                    assert len(frame_list) == self.sequence_len
                    sequence_cube_vgg = []

                    for frame_path in frame_list:
                        image_vgg = self.load_image_vgg(os.path.join(self.dataset_dir, folder, instance,frame_path))
                        sequence_cube_vgg.append(image_vgg)

                    sequence_cube_vgg = np.stack(sequence_cube_vgg, axis = 0)
                    sequence_cube_vgg = self.preprocess_image_vgg(sequence_cube_vgg)

                    if frame_list[-1].endswith('.jpg'):
                        cube_name = (frame_list[-1].replace('.jpg', ''))
                    else:
                        cube_name = (frame_list[-1].replace('.png', ''))

                    sequence_cube_name_vgg = os.path.join(self.dataset_dir, folder, instance, cube_name + "_vgg.npy")
                    np.save(sequence_cube_name_vgg, sequence_cube_vgg)

    def preprocess_folder_content(self, folder, particular_batch_size):
        all_image_paths = []
        for root, dirs, files in os.walk(os.path.join(self.dataset_dir, folder)):
            for file in files:
                if file.endswith(self.file_ending):
                    all_image_paths.append(os.path.join(root, file))
        labels = [(int(x.split('__')[2])) for x in all_image_paths]

        images_tensor_slices = tf.data.Dataset.from_tensor_slices(all_image_paths)
        cubes = images_tensor_slices.map(lambda x: tf.numpy_function(self.load_np_cube, [x], tf.float32),
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)

        labels = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))

        dataset = tf.data.Dataset.zip((cubes, labels))
        dataset = dataset.shuffle(1000).batch(particular_batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def display_sequences_train(self):
        for temporal_batch, labels in self.train_ds:
            for batch in temporal_batch:
                for image in batch:
                    if self.base_type == 'VGG':
                        cv2.imshow('test', cv2.cvtColor(np.array(image) / 255, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(100)
                cv2.destroyAllWindows()

    def hmdb_int2label(self, int_label):

        dictionary = {"0": "smoke", "1": "fall_floor", "2": "throw", "3": "kiss", "4": "push", "5": "pullup",
                      "6": "pushup", "7": "run", "8": "chew", "9": "cartwheel", "10": "ride_horse", "11": "brush_hair",
                      "12": "stand", "13": "laugh", "14": "kick", "15": "dribble", "16": "walk", "17": "climb_stairs",
                      "18": "shoot_gun", "19": "ride_bike", "20": "pour", "21": "catch", "22": "draw_sword", "23": "shoot_ball",
                      "24": "talk", "25": "sword_exercise", "26": "kick_ball", "27": "jump", "28": "drink", "29": "handstand",
                      "30": "sit", "31": "eat", "32": "hug", "33": "hit", "34": "shoot_bow", "35": "pick", "36": "climb",
                      "37": "dive", "38": "clap", "39": "shake_hands", "40": "wave", "41": "swing_baseball", "42": "golf",
                      "43": "flic_flac", "44": "fencing", "45": "somersault", "46": "punch", "47": "situp", "48": "sword",
                      "49": "turn", "50": "smile"}

        label = dictionary[str(int_label)]
        return label



    def display_sequences_test(self):
        for temporal_batch, labels in self.test_ds:
            for batch in temporal_batch:
                for image in batch:
                    if self.base_type == 'VGG':
                        cv2.imshow('test', cv2.cvtColor(np.array(image) / 255, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(100)
                cv2.destroyAllWindows()

    def get_datasets_sequence(self):
        self.save_4d_to_numpy()
        self.train_ds = self.preprocess_folder_content('train', self.batch_size)
        self.test_ds = self.preprocess_folder_content('test',self.batch_size)
        self.val_ds = self.preprocess_folder_content('val', self.batch_size)

        return self.train_ds, self.test_ds, self.val_ds




