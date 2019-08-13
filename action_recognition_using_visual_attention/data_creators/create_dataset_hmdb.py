import json
import os
import cv2
import numpy as np
import shutil
import pandas as pd
from preprocess_hmdb_splits import preprocess_splits


def create_hmdb_dataset(directory,
                        test_split,
                        val_split,
                        set_fps,
                        sequence_len):

    preprocess_splits(directory)
    directory_videos = os.path.join(directory,'HMDB_videos')
    directory_save = os.path.join(directory, 'HMDB_dataset')
    data = pd.read_csv(os.path.join(directory, 'all_splits.txt'), sep=" ", header=None)
    data.columns = ["video", "label", "0"]

    assert val_split + test_split == 1

    dict_int_to_class = dict()
    class_integer = 0
    instance_integer = 0
    try:
        shutil.rmtree(directory_save)
    except:
        pass
    try:
        folders = ['train', 'test', 'val']
        os.mkdir(directory_save)
        for i in folders:
            os.mkdir(os.path.join(directory_save, i))
    except:
        pass
    print('Creating the dataset...')
    for j, folder in enumerate(os.listdir(directory_videos)):
        print('Processing class {} ({} out of 51)'.format(folder,j+1))
        dict_int_to_class[class_integer] = folder
        for movie in os.listdir(os.path.join(directory_videos, folder)):
            which_split = int(data["label"][data['video'] == movie])
            if which_split == 0 or which_split == 1:
                split = 'train'
            else:
                split = np.random.choice(['val', 'test'], 1, p=[val_split, test_split])
                split = split[0]

            vidcap = cv2.VideoCapture(os.path.join(directory_videos, folder, movie))

            fps = vidcap.get(cv2.CAP_PROP_FPS)

            skip = np.floor(fps/set_fps)
            count = 0
            ret = True
            while vidcap.isOpened() and ret == True:
                ret, frame = vidcap.read()

                if (count % skip ==0) and (type(frame) is np.ndarray):
                    try:
                        os.mkdir(os.path.join(directory_save, split, str(instance_integer)))
                    except:
                        pass
                    cv2.imwrite(os.path.join(directory_save, split, str(instance_integer), "__{:03d}__{}__.jpg".format(count, str(class_integer))), frame)

                count = count + 1

                if count == sequence_len * skip:
                    count = 0
                    instance_integer = instance_integer + 1

            instance_integer = instance_integer + 1

        class_integer = class_integer + 1
        with open(os.path.join(directory_save, "dict.json"), 'w') as fp:
            json.dump(dict_int_to_class, fp)


    for folder in ['train', 'test', 'val']:

        list_folders = os.listdir(os.path.join(directory_save, folder))

        for folder_1 in list_folders:
            folder_name = os.path.join(directory_save, folder, folder_1)
            if len(os.listdir(folder_name)) == sequence_len:
                pass
            else:
                shutil.rmtree(folder_name)

if __name__ == "__main__":

    ###### Parameters of the dataset ######
    directory = "ROOT DIR OF YOUR PROJECT"
    no_videos_test = 400
    test_split = no_videos_test / 1530
    val_split = 1 - test_split
    set_fps = 15
    sequence_len = 30
    #######################################

    create_hmdb_dataset(directory,
                        test_split,
                        val_split,
                        set_fps,
                        sequence_len)








