import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import ImageDraw, ImageFont, Image
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter


class SequentialInference:
    def __init__(self, config_data):
        self.test_directory = config_data['test_dir']
        self.save_dir = config_data['prediction_save_dir']
        self.batch_no = 0
        self.image_size = 500

    def save_image(self, image, attention_map, prediction, label, i):
        if i == 0:
            self.batch_no = self.batch_no + 1

        batch_shape = image.shape[0]

        for j in range(batch_shape):
            one_image = image[j, :, :, :]
            one_attention_map = attention_map[j, :, :]
            one_prediction = prediction[j, :]
            one_label = label[j]

            original_image = one_image.numpy().reshape((image.shape[1], image.shape[2], image.shape[3]))
            original_image = imresize(original_image, (self.image_size, self.image_size))
            score_map = one_attention_map.numpy().reshape((attention_map.shape[1], attention_map.shape[2]))
            score_map = imresize(score_map, interp='cubic', size=(self.image_size, self.image_size))
            score_map = gaussian_filter(score_map, sigma=30)
            percentile = np.percentile(score_map, 80)
            score_map = np.where(score_map > percentile, score_map, 0)
            image_prediction = np.argmax(one_prediction)
            confidence = one_prediction[image_prediction].numpy()
            image_label = int(one_label.numpy())
            plt.axis('off')
            self.save_single_image(original_image, score_map, image_label, image_prediction, confidence, i, j)

    def save_single_image(self, image, attention, label, decision, confidence, i, j):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        attention = cv2.cvtColor(attention, cv2.COLOR_GRAY2BGR)
        attention = cv2.applyColorMap(attention, cv2.COLORMAP_JET)
        attention = Image.fromarray(attention)
        result = Image.blend(image, attention, alpha=0.5)

        draw = ImageDraw.Draw(result)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20, encoding="unic")
        except:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 20, encoding="unic")

        draw.text((50, 50), 'label: {} \n pred: {} ({:.2f})'.format(str(self.hmdb_int2label(label)),
                                                                    str(self.hmdb_int2label(decision)),
                                                                    confidence), (255, 255, 255),
                  font=font)

        save_directory = os.path.join(self.save_dir, '{}-{}-{}.png'.format(self.batch_no, j, i))
        result.save(save_directory)
        plt.clf()

    def hmdb_int2label(self, int_label):
            dictionary = {"0": "golf", "1": "situp", "2": "wave", "3": "dribble", "4": "turn", "5": "walk", "6": "pour", "7": "kiss", "8": "smile", "9": "hug", "10": "stand", "11": "flic_flac", "12": "eat", "13": "throw", "14": "climb", "15": "pullup", "16": "punch", "17": "laugh", "18": "brush_hair", "19": "somersault", "20": "sit", "21": "swing_baseball", "22": "kick_ball", "23": "shake_hands", "24": "shoot_bow", "25": "fall_floor", "26": "run", "27": "ride_bike", "28": "jump", "29": "drink", "30": "sword_exercise", "31": "shoot_gun", "32": "ride_horse", "33": "clap", "34": "shoot_ball", "35": "cartwheel", "36": "hit", "37": "push", "38": "kick", "39": "pick", "40": "pushup", "41": "sword", "42": "talk", "43": "draw_sword", "44": "chew", "45": "dive", "46": "smoke", "47": "fencing", "48": "climb_stairs", "49": "catch", "50": "handstand"}

            label = dictionary[str(int_label)]
            return label





