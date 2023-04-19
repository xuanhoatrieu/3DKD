import numpy as np
import os
import csv
import cv2
import random
import glob
import json
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import BaseLogger

def check_label(start_frame, end_frame, start_falling, end_falling):
    mid_frame = (start_frame + end_frame) // 2
    if start_falling < mid_frame and mid_frame < end_falling:
        return 1
    else:
        return 0

def video_split(filepath, num_frames_per_clip):
    new_data = []
    df = pd.read_csv(filepath)
    folder_path = df['folder_path'].to_list()
    num_frame = df['num_frames'].to_list()
    falling_frame = df['falling_frame'].to_list()
    num_video = len(folder_path)
    for video in range(num_video):
        frames = num_frame[video]
        if frames % num_frames_per_clip ==0:
            frames-=1
        num_clips = frames // num_frames_per_clip
        path = folder_path[video] + '-cam0-rgb'
        start_falling = falling_frame[video]
        end_falling = frames
        for i in range(num_clips):
            clip = {}
            start_frame = i * num_frames_per_clip
            end_frame = (i+1) * num_frames_per_clip
            if path[:3] =='adl':
                label=0
            else:
                label=check_label(start_frame,end_frame,start_falling, end_falling)
            clip['video_path'] = path
            clip['start_frame'] = start_frame
            clip['end_frame'] = end_frame
            clip['label'] = label
            new_data.append(clip)
        if path[:4]== 'fall':
            for i in range(start_falling, end_falling - num_frames_per_clip):
                clip = {}
                start_frame = i
                end_frame = start_frame + num_frames_per_clip
                label = 1
                clip['video_path'] = path
                clip['start_frame'] = start_frame
                clip['end_frame'] = end_frame
                clip['label'] = label
                new_data.append(clip)
    return new_data


def count(data):
    no_falling = 0
    falling = 0
    for d in data:
        if d['label']==0:
            no_falling+=1
        else:
            falling+=1
    return falling, no_falling

def split_train_test_data(full_data):
    data_train, data_test = train_test_split(full_data, test_size=0.2, random_state=43)
    with open("train.csv", "w") as file:
        for data in data_train:
            s = data['video_path'] + ',' + str(data['start_frame']) + ',' + str(data['end_frame']) + ',' + str(data['label']) + '\n'
            file.writelines(s)
        file.close()

    with open("test.csv", "w") as file:
        for data in data_test:
            s = data['video_path'] + ',' + str(data['start_frame']) + ',' + str(data['end_frame']) + ',' + str(data['label'])+ '\n'
            file.writelines(s)
        file.close()

def get_data(csv_file):
    """Load our data from file."""
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    return data

def get_frames_for_sample(sample, num_frames_per_clip):
    """Given a sample row from the data file, get all the corresponding frame
    filenames."""
    folder_name = sample[0].decode('UTF-8')
    # images = sorted(glob.glob(os.path.join(folder_name, '/*jpg')))
    images = sorted(glob.glob(os.path.join('dataset', folder_name + '/*png')))
    # images = sorted(glob.glob(folder_name + '/*jpg'))
    start_idx = int(sample[1])
    label = int(sample[3])
    return images, start_idx, label


# def read_imgs(start_frame, end_frame, rgb_folder):
#     rgb_clip = []
#     for i in range(start_frame, end_frame):
#         rgb_file = os.path.join('dataset', rgb_folder, rgb_folder + '-' + str(i + 1).zfill(3) + ".png")
#         rgb_img = Image.open(rgb_file)
#         rgb_img = rgb_img.resize((224,224))
#         rgb_img = np.asarray(rgb_img) / 255.0
#         rgb_clip.append(rgb_img)

#     return rgb_clip


# def read_images(frames, start_idx, num_frames_per_clip):
#     img_data = []
#     if (len(frames)< start_idx-num_frames_per_clip):
#         print(frames[0])
#     for i in range(start_idx, start_idx + num_frames_per_clip):
#         if i>= len(frames):
#             print(frames[i-1])
#         img = Image.open(frames[i])
#         img = np.asarray(img)
#         img_data.append(img)
#     return img_data

def read_images(frames, start_idx, num_frames_per_clip):
    img_data = []
    for i in range(start_idx, start_idx + num_frames_per_clip):
        img = Image.open(frames[i])
        img = np.asarray(img)
        img_data.append(img)
    return img_data

def data_process(tmp_data, crop_size, is_train):
    img_datas = []
    crop_x = 0
    crop_y = 0

    if crop_size==224:
        resize_value=256
    else:
        resize_value=129

    if is_train and random.random()>0.5:
        flip = True
    else:
        flip = False

    if is_train and random.random()>0.8:
        cvt_color = True
    else:
        cvt_color = False

    if is_train and random.random()>0.5:
        channel1, channel2 = random.choices([0, 1, 2], k=2)
    else:
        channel1, channel2 = 0, 0

    size = crop_size
    if is_train and crop_size==112:
        size = random.choice([129, 112, 96, 84])
        # size = random.choice([129, 112, 96])

    if is_train and crop_size==224:
        size = random.choice([256, 224, 192, 168])
        # size = random.choice([256, 224, 192])

    for j in range(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(resize_value) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), resize_value))).astype(np.float32)
        else:
            scale = float(resize_value) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (resize_value, int(img.height * scale + 1)))).astype(np.float32)
        if j == 0:
            if is_train:
                crop_x = random.randint(0, int(img.shape[0] - size))
                crop_y = random.randint(0, int(img.shape[1] - size))
            else:
                crop_x = int((img.shape[0] - size) / 2)
                crop_y = int((img.shape[1] - size) / 2)
        img = img[crop_x:crop_x + size, crop_y:crop_y + size, :]
        img = np.array(cv2.resize(img, (crop_size, crop_size))).astype(np.float32)
        img = np.asarray(img) / 127.5
        img -= 1.

        if flip:
            img = np.flip(img, axis=1)

        if cvt_color:
            img = -img

        if channel1 != channel2:
            img = Channel_splitting(img, channel1, channel2)

        img_datas.append(img)
    return img_datas

def generator_training_data(data, num_frames_per_clip=16, crop_size=224):
    while True:
        np.random.shuffle(data)
        for i in range(len(data)):
            row = data[i]
            frames, start_idx, label = get_frames_for_sample(row, num_frames_per_clip)  # read all frames in video and length of the video
            clip = read_images(frames, start_idx, num_frames_per_clip)
            clip1 = data_process(clip, crop_size, is_train=True)
            # clip2 = data_process(clip, crop_size, is_train=True)
            clip1 = np.asarray(clip1)
            # clip2 = np.asarray(clip2)
            yield clip1, label

def generator_test_data(data, num_frames_per_clip=16, crop_size=224):
    while True:
        # np.random.shuffle(data)
        for i in range(len(data)):
            row = data[i]
            frames, start_idx, label = get_frames_for_sample(row, num_frames_per_clip)  # read all frames in video and length of the video
            clip = read_images(frames, start_idx, num_frames_per_clip)
            clip = data_process(clip, crop_size, is_train=False)
            clip = np.asarray(clip)
            yield clip, label

def adjust_constrast_and_brightness(clip, alpha, beta):
    clip = clip * alpha + beta
    return clip

def add_noise_clip(clip, stddev_value=0.1):
    noise_clip = tf.random.normal(shape=clip.shape, mean=0, stddev=stddev_value)
    # noise_clip = np.random.normal(loc=0, scale=stddev_value, size=clip.shape)
    return clip + noise_clip

def Channel_splitting(clip, channel1, channel2):
    clip[..., channel1] = clip[...,channel2]
    return clip

def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
    radius = tf.compat.v1.to_int32(kernel_size / 2)
    kernel_size = radius * 2 + 1
    x = tf.compat.v1.to_float(tf.range(-radius, radius + 1))
    blur_filter = tf.exp(-tf.pow(x, 2.0) / (2.0 * tf.pow(tf.compat.v1.to_float(sigma), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred


def adjust_hue(clip, delta):
    clip = tf.image.adjust_hue(clip, delta=delta)
    return clip

def data_augmentation(clip1, label):
    if random.random() > 0.5:
        alpha = random.uniform(0.5, 1.5)
        beta = random.uniform(-0.5, 0.5)
        clip1 = adjust_constrast_and_brightness(clip1, alpha, beta)
        clip1 = tf.clip_by_value(clip1, -1., 1.)

    if random.random() > 0.5:
        sigma = random.uniform(0.1, 2.0)
        clip1 = gaussian_blur(clip1, kernel_size=7, sigma=sigma)
        clip1 = tf.clip_by_value(clip1, -1., 1.)

    if random.random() > 0.5:
        hue_value = random.uniform(0, 0.1)
        clip1 = adjust_hue(clip1, hue_value)
        clip1 = tf.clip_by_value(clip1, -1., 1.)

    # if random.random() > 0.5:
    #     alpha = random.uniform(0.5, 1.5)
    #     beta = random.uniform(-0.5, 0.5)
    #     clip2 = adjust_constrast_and_brightness(clip2, alpha, beta)
    #     clip2 = tf.clip_by_value(clip2, -1., 1.)

    # if random.random() > 0.5:
    #     sigma = random.uniform(0.1, 2.0)
    #     clip2 = gaussian_blur(clip2, kernel_size=7, sigma=sigma)
    #     clip2 = tf.clip_by_value(clip2, -1., 1.)

    # if random.random() > 0.5:
    #     hue_value = random.uniform(0, 0.1)
    #     clip2 = adjust_hue(clip2, hue_value)
    #     clip2 = tf.clip_by_value(clip2, -1., 1.)
    return clip1, label

class ModelCheckpoint(BaseLogger):
    def __init__(self, folderpath, jsonPath=None, jsonName=None, monitor='val_accuracy', mode='max', verbose=1):
        super(ModelCheckpoint, self).__init__()
        self.filepath = os.path.join(folderpath, 'best_model.h5')
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.H = {}
        self.jsonPath = jsonPath
        self.jsonName = jsonName
        self.jsonfile = os.path.join(self.jsonPath, self.jsonName)

        if self.mode == 'max':
            self.monitor_op = np.greater
            self.current_best = -np.Inf
        else:
            self.monitor_op = np.less
            self.current_best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        self.model.accuracy_s.reset_states()
        self.model.accuracy_t.reset_states()
        self.model.val_accuracy.reset_states()
        self.model.Sensitivity.reset_states()
        self.model.Specificity.reset_states()

        current = logs.get(self.monitor)
        lr = K.get_value(self.model.optimizer.lr)
        print('\nCurrent best accuracy: ', self.current_best)
        print('Current LR: %0.5f' % (lr))

        if self.monitor_op(current, self.current_best):
            if self.verbose > 0:
                print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s'
                      % (epoch + 1, self.monitor, self.current_best, current, self.filepath))

            self.model.student_model.save_weights(self.filepath, overwrite=True)
            self.current_best = current
        
        # for (k, v) in logs.items():
        #     l = self.H.get(k, [])
        #     print(k, v)
        #     if len(v)>1:
        #         v = np.mean(v)
        #     l.append(float(v))
        #     self.H[k] = l
        #
        # # check to see if the training history should be serialized
        # # to file
        # if self.jsonfile is not None:
        #     f = open(self.jsonfile, "w")
        #     f.write(json.dumps(self.H))
        #     f.close()



# data = get_data('full.csv')
# new_data= video_split('full.csv', 16)
# split_train_test_data(new_data)
