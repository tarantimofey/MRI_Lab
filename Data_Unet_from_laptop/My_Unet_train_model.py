#C:/Users/Taran/AppData/Local/Programs/Python/Python39/python.exe -i "$(FULL_CURRENT_PATH)"

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math 
#import My_Preful_lib.My_Preful_lib as mp
from os import listdir
from os.path import isfile, join
import cv2
import imageio

import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import random
from tqdm import tqdm

import pltable
from pltable import PrettyTable

from skimage.io import imread, imshow
from skimage.transform import resize
import time
start_time = time.time()
print('Programm start, time:', start_time)



IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1


PATH = []
"""
PATH.append('C:/Users/Taran/Desktop/Ready_data/OP_NS/Png/')
PATH.append('C:/Users/Taran/Desktop/Ready_data/OP_SS/Png/')
PATH.append('C:/Users/Taran/Desktop/Ready_data/OP_15/Png/')
PATH.append('C:/Users/Taran/Desktop/Ready_data/OP_NS_1800/Png/')
PATH.append('C:/Users/Taran/Desktop/Ready_data/olyafirst1200/Png/')
"""

"""
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/5/5_1/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/5/5_2/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/5/5_3/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/5/5_4/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/5/5_5/Png')

PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/7/7_1/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/7/7_2/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/7/7_3/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/7/7_4/Png')

PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/9/9_1/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/9/9_2/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/9/9_3/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/9/9_4/Png')

PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/10/10_1/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/10/10_2/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/10/10_3/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/10/10_4/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/10/10_5/Png')

PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/12/12_1/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/12/12_2/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/12/12_3/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/mct/12/12_4/Png')

PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/BLEOMICINE/13rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/BLEOMICINE/22rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/BLEOMICINE/23rat_bad/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/BLEOMICINE/24rat_bad/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/BLEOMICINE/26rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/BLEOMICINE/27rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/BLEOMICINE/28rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/BLEOMICINE/29rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/BLEOMICINE/30rat/Png')
"""
"""
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/CONTROL/1rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/CONTROL/2rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/CONTROL/3rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/CONTROL/10rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/CONTROL/11rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/CONTROL/12rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/CONTROL/15rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/CONTROL/19rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/CONTROL/20rat/Png')
PATH.append('C:/Users/Taran/Desktop/Data_Unet/Data/Rat/BLEOM/CONTROL/21rat_bad/Png')
"""





"""
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/1rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/2rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/3rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/10rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/11rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/12rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/15rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/19rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/20rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/21rat/Png')
"""
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/22rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/23rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/24rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/26rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/27rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/28rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/29rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/30rat/Png')

PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/4rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/5rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/7rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/8rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/9rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/14rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/16rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/17rat/Png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/18rat/Png')










model_path = 'C:/Users/Taran/Desktop/Data_Unet/Models/Olya_26_07_23/PAT/'
#model_path = 'C:/Users/Taran/Desktop/Data_Unet/Models/mct/edema/1Run/'
#model_file = model_path + 'MRI_lung'
model_file = model_path + 'Pat'


image_folder = '/EXP_DATA/'
#mask_folder = '/Mask_edema/'
mask_folder = '/MASK_PAT/'

folder_2_flag = 0
image_folder_2 = '/INSP_DATA/'
mask_folder_2 = '/MASK_INSP/'

check_folders_flag = 0
test_split_coeff = 0.02

model_type = 2
batch_size = 32





"""
PATH.append('C:/Users/Taran/Desktop/rat_5/LUNGS.h64/2/pdata/1/png/XY') #0
PATH.append('C:/Users/Taran/Desktop/rat_5/LUNGS.ha4/3/pdata/1/png/XY') #1
PATH.append('C:/Users/Taran/Desktop/rat_5/LUNGS.hh3/2/pdata/1/png/XY') #2
PATH.append('C:/Users/Taran/Desktop/rat_5/LUNGS.hh3/3/pdata/1/png/XY') #3
PATH.append('C:/Users/Taran/Desktop/rat_5/LUNGS.ho3/3/pdata/1/png/XY') #4
#PATH.append('C:/Users/Taran/Desktop/rat_5/LUNGS.ho3/4/pdata/1/png/XY') #5

#model_file = 'C:/Users/Taran/Desktop/Data_Unet/Models/rat5/1Run/rat5_6.h5'
"""


if not os.path.exists(model_path):
    os.makedirs(model_path)


##########################################################################
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


if model_type == 0:
    #model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/MRI_lung_0.h5')
    model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/2Run/MRI_lung_SS_15.h5')
if model_type == 1:
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x/1.0)(inputs)


    #Contracting path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

    c3 = tf.keras.layers.Conv2D(63, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(63, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)


    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.2)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expanding path
    u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.2)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.2)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.summary()
if model_type == 2:
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x/1.0)(inputs)


    #Contracting path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.2)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    p5 = tf.keras.layers.MaxPooling2D((2,2))(c5)


    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    #Expanding path
    u7 = tf.keras.layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c5])
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c4])
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.2)(c8)
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c3])
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.2)(c9)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    u10 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c9)
    u10 = tf.keras.layers.concatenate([u10, c2])
    c10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
    c10 = tf.keras.layers.Dropout(0.2)(c10)
    c10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)
    
    u11 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c10)
    u11 = tf.keras.layers.concatenate([u11, c1])
    c11 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
    c11 = tf.keras.layers.Dropout(0.2)(c11)
    c11 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)

    outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c11)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.summary()
if model_type == 3:
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x/1.0)(inputs)


    #Contracting path
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)


    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.2)(c5)
    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expanding path
    u6 = tf.keras.layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.2)(c8)
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.2)(c9)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.summary()

###################################################################################



"""
def Show_images(n):

    X_img = X_test[n][:, :, 0].astype(np.uint8)
    Y_img = Y_test[n][:, :, 0].astype(np.uint8)
    Y_ground_img = Y_ground_test[n][:, :, 0].astype(np.uint8)

    #X_img = X_img*255
    #Y_img = Y_img*255
    X_img = cv2.normalize(X_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img = cv2.normalize(Y_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_ground_img = cv2.normalize(Y_ground_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    cv2.namedWindow('X', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X", X_img)
    #cv2.resizeWindow('X', 200, 200)
    cv2.namedWindow('Y', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Y", Y_img)
    #cv2.resizeWindow('Y', 200, 200)
    cv2.namedWindow('Y_ground', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Y_ground", Y_ground_img)
    #cv2.resizeWindow('Y', 200, 200)


    Y_img_color = Y_img.copy()
    Y_img_color = cv2.cvtColor(Y_img_color, cv2.COLOR_GRAY2RGB)

    for i in range(0, len(Y_img_color)):
        for j in range(0, len(Y_img_color)):
            Y_img_color[i][j][0] = 0
            Y_img_color[i][j][1] = 0       

    X_img_color = X_img.copy()
    X_img_color = cv2.cvtColor(X_img_color, cv2.COLOR_GRAY2RGB)
    global superimposed
    superimposed  = cv2.addWeighted(X_img_color, 1, Y_img_color, 0.5, 0)

    cv2.namedWindow('Img+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Img+mask", superimposed)
"""

def Show_images(n):

    global X_img
    global Y_img
    global Y_img_refined
    global Y_img_ground
    global X_masked
    global X_thresholded
    
    X_img = X_test[n][:, :, 0].astype(np.uint8)
    Y_img = Y_test[n][:, :, 0].astype(np.uint8)
    # Y_img_refined = Y_refined[n][:, :].astype(np.uint8)
    Y_img_ground = Y_ground[n]
    # X_img_masked = X_masked[n][:, :].astype(np.uint8)
    # X_img_thresholded = X_thresholded[n]
        
    X_img = cv2.normalize(X_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img = cv2.normalize(Y_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img_ground = cv2.normalize(Y_img_ground, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img_ground *= 255
    #Y_img_refined = cv2.normalize(Y_img_refined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # X_img_masked = cv2.normalize(X_img_masked, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # X_img_thresholded = cv2.normalize(X_img_thresholded, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    cv2.namedWindow('X', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X", X_img)
    #cv2.resizeWindow('X', 200, 200)
    cv2.namedWindow('Y', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Y", Y_img)
    #cv2.resizeWindow('Y', 200, 200)
    # cv2.namedWindow('Y_refined', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("Y_refined", Y_img_refined)
    #cv2.resizeWindow('Y', 200, 200)
    cv2.namedWindow('Y_ground', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Y_ground", Y_img_ground)
    #cv2.resizeWindow('Y', 200, 200)
    # cv2.namedWindow('X_masked', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("X_masked", X_img_masked)
    #cv2.resizeWindow('Y', 200, 200)
    # cv2.namedWindow('X_thresholded', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("X_thresholded", X_img_thresholded)
    #cv2.resizeWindow('Y', 200, 200)


    global superimposed
    superimposed  = Superimpose(X_img, Y_img)
    cv2.namedWindow('Img+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Img+mask", superimposed)

    # global superimposed_refined
    # superimposed_refined  = Superimpose(X_img, Y_img_refined)
    # cv2.namedWindow('Img+mask_refined', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("Img+mask_refined", superimposed_refined)

    global superimposed_ground
    superimposed_ground  = Superimpose(X_img, Y_img_ground)
    cv2.namedWindow('Img+mask_ground', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Img+mask_ground", superimposed_ground)
    
    print('\nDice_coeff = ', Dice_coeff(Y_img, Y_img_ground))


def Superimpose(img1, img2):
    img2_color = img2.copy().astype(np.uint8)
    img2_color = cv2.cvtColor(img2_color, cv2.COLOR_GRAY2RGB)

    img2_color[:,:,0] = 0
    img2_color[:,:,1] = 0

    img1_color = img1.copy()
    img1_color = cv2.cvtColor(img1_color, cv2.COLOR_GRAY2RGB)
    superimposed = cv2.addWeighted(img1_color, 1, img2_color, 0.5, 0)
    return superimposed
    
def Dice_coeff(gt, seg):
    k = seg.max()
    dice = np.sum(seg[gt==k]==k)*2.0 / (np.sum(seg[seg==k]==k) + np.sum(gt[gt==k]==k))
    return dice


image_ids = []
for i, path in enumerate(PATH):
    temp = listdir(path + mask_folder)
    temp.sort(key=len)
    image_ids.append(temp)

"""
image_mask_ids = []
for i, path in enumerate(PATH):
    temp = listdir(path + image_folder)
    temp.sort(key=len)
    image_ids.append(temp)
"""
if folder_2_flag == 1:
    image_ids_2 = []
    for i, path in enumerate(PATH):
        temp = listdir(path + mask_folder_2)
        temp.sort(key=len)
        image_ids_2.append(temp)

"""
mask_ids = []
for i, path in enumerate(PATH):
    temp = listdir(path + mask_folder)
    temp.sort(key=len)
    mask_ids.append(temp)
if len(mask_ids) != len(image_ids):
    raise ValueError('Image and mask sequences have different lenght')
"""
"""
image_ids = listdir(PATH + '/Image')
image_ids.sort(key=len)
"""
#print(image_ids)
"""
X = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint32)
Y = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint32)
"""
X = []
Y = []

print('\nResizing training images...')
for i, path in enumerate(PATH):
    print('Dataset ' + str(i) + ':')
    print(PATH[i] + '   ' + image_folder + '<--->' + mask_folder)
    for n, id_ in tqdm(enumerate(image_ids[i]), total=len(image_ids[i])):
        #path = PATH
        img = imread(path + image_folder + id_)[:,:]
        image_mask_folder = '/Mask/'
        #img_mask = imread(path + image_mask_folder + id_)[:,:]
        #img = cv2.bitwise_or(img, img, mask=img_mask)
        img = img[..., np.newaxis]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        #img = img.astype(float)/np.max(img)
        
        
        #img_mask = img_mask[..., np.newaxis]
        #img_mask = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        
        #print(img.size, img_mask.size)
        #img = cv2.bitwise_or(img, img, mask=img_mask)
        X.append(img)
            
        mask = imread(path + mask_folder + id_)[:,:]
        #mask = cv2.bitwise_or(mask, mask, mask=img_mask)
        mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        mask = mask[..., np.newaxis]
        mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        Y.append(mask)

if folder_2_flag == 1:
    print('\nResizing training images in folder 2...')
    for i, path in enumerate(PATH):
        print('Dataset ' + str(i) + ':')
        print(PATH[i] + '   ' + image_folder_2 + '<--->' + mask_folder_2)
        for n, id_ in tqdm(enumerate(image_ids_2[i]), total=len(image_ids_2[i])):
            #path = PATH
            img = imread(path + image_folder_2 + id_)[:,:]
            img = img[..., np.newaxis]
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            #img = img.astype(float)/np.max(img)
            X.append(img)
                
            mask = imread(path + mask_folder_2 + id_)[:,:]
            mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
            mask = mask[..., np.newaxis]
            mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            Y.append(mask)


print('Done!\n')
X = np.asarray(X)
Y = np.asarray(Y)
print(str(len(X)) + ' images were found\n')

"""
with open('K:/X_data.txt', 'wb') as f:
    f.write(X)
with open('K:/Y_data.txt', 'wb') as f:
    f.write(Y)
"""

if check_folders_flag == 1:
    raise SystemExit

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split_coeff, random_state=1, shuffle='True')
print(str(len(X_train)) + ' images selected for training (' + str(test_split_coeff*100) + '%)\n\n')
Y_ground = Y_test.copy()
checkpointer = tf.keras.callbacks.ModelCheckpoint(model_file+'.h5', verbose=1, save_best_only='True')
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir=model_path)
    #tf.keras.callbacks.LearningRateScheduler(scheduler)
]

print('Fitting model...')
#with tf.device('/GPU:1'):
results = model.fit(X_train,Y_train, validation_split=0.1, batch_size=batch_size, epochs=100, callbacks=[callbacks, checkpointer])
print('Done!\n')

print('Fitting test data...')
Y_test = model.predict(X_test, verbose=1)
Y_test = (Y_test > 0.5).astype(np.uint32)
Y_test *= 255
print('Done!\n\n')


dice_table = []
for i in range(len(Y_test)):
    Y_img = Y_test[i][:, :, 0].astype(np.uint8)
    Y_img_ground = Y_ground[i]
    Y_img = cv2.normalize(Y_img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img_ground = cv2.normalize(Y_img_ground, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    d = Dice_coeff(Y_img, Y_img_ground)
    dice_table.append(d)
dice_table = np.asarray(dice_table)
print('Dice_coefficient:\nmean = ', dice_table.mean(), '\nmax = ', dice_table.max(), '\nmin = ', dice_table.min(), '\n\n')


end_time = time.time()
print('Execution time:', end_time - start_time, 's')


#Path_table.field_names = ['N folder', 'floder name', 'N of images']


txt_file = open(model_file+'.txt', 'w', encoding="utf-8")
txt_file.writelines('-----------------------------------------------------------------\n') 
for i in range(len(PATH)):
    txt_file.writelines('Dataset ' + str(i) + ':')
    txt_file.writelines('\npath: ' + PATH[i] + '\n')
    Path_table = PrettyTable()
    Path_table.set_style(pltable.UNICODE_LINES) 
    #Path_table.clear()
    Path_table.field_names = ['N folder', 'folder name', 'N of images']
    Path_table.add_row([1, image_folder + '<--->' + mask_folder, str(len(image_ids[i]))])
    #txt_file.writelines('\nfolder_1: ' + image_folder + '<--->' + mask_folder + '; N_images: ' + str(len(image_ids[i])))
    if folder_2_flag == 1:
        #txt_file.writelines('\nfolder_2: ' + image_folder_2 + '<--->' + mask_folder_2 + '; N_images: ' + str(len(image_ids_2[i])))
        Path_table.add_row([2, image_folder_2 + '<--->' + mask_folder_2, str(len(image_ids_2[i]))])
    #txt_file.writelines(Path_table.get_string())
    txt_file.writelines(str(Path_table))
    txt_file.writelines('\n\n')
txt_file.writelines('Total N of images: ' + str(len(X)) + '\n')
txt_file.writelines('N of train images: ' + str(len(X_train)) + '\n')
txt_file.writelines('N of test images: ' + str(len(X_test)) + '\n\n')
#txt_file.writelines(str(model.get_metrics_result()))   
txt_file.writelines('-----------------------------------------------------------------')    
txt_file.writelines('\nBatch size: ' + str(batch_size))    
txt_file.writelines('\nN of epochs: ' + str(len(results.history['accuracy'])-4))    
txt_file.writelines('\naccuracy: ' + str(results.history['accuracy'][-1]))    
txt_file.writelines('\nval_accuracy: ' + str(results.history['val_accuracy'][-1]))    
txt_file.writelines('\nloss: ' + str(results.history['loss'][-1]))    
txt_file.writelines('\nval_loss: ' + str(results.history['val_loss'][-1]))  
txt_file.writelines('\n\n-----------------------------------------------------------------')   
txt_file.writelines('\nDice_coefficient:\nmean = ' + str(dice_table.mean()) + '\nmax = ' + str(dice_table.max()) + '\nmin = ' + str(dice_table.min())) 
txt_file.writelines('\n\n-----------------------------------------------------------------')   
txt_file.writelines('\nExecution time: ' + str(round((end_time - start_time), 5)) + 's')  
txt_file.writelines('\n\n-----------------------------------------------------------------')
txt_file.close()

with open(model_file+'_summary.txt', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
with open(model_file+'_dice.txt', 'w') as f:
    for i in range(len(dice_table)):
        f.writelines(str(dice_table[i]) + ', ')

metrics_file = open(model_file+'_metrics.txt', 'w')
metrics_file.writelines('accuracy = ' + str(results.history['accuracy']))    
metrics_file.writelines('\n\nval_accuracy = ' + str(results.history['val_accuracy']))    
metrics_file.writelines('\n\nloss = ' + str(results.history['loss']))    
metrics_file.writelines('\n\nval_loss = ' + str(results.history['val_loss']))    
metrics_file.close()


# plot the training process
plt.figure(figsize=[6, 4])
plt.subplot(1, 2, 1)
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.xlabel('epoch')
plt.title('accuracy')
plt.ylabel('accuracy')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.savefig(model_path + 'accuracy_loss_graph.png')
plt.savefig(model_path + 'accuracy_loss_graph.pdf')
plt.show(block=False)


#print(results.history)





current_img = 0
Show_images(current_img)

img_len = len(X_test)
while(1):
    full_key_code = cv2.waitKeyEx(0)
    if full_key_code == 2555904:
        current_img += 1
        if current_img >= img_len:
            current_img = img_len-1
        Show_images(current_img)
        print('image ' + str(current_img) + ' / ' + str(img_len))
    if full_key_code == 2424832:
        current_img -= 1        
        if current_img < 0:
            current_img = 0
        Show_images(current_img)
        print('image ' + str(current_img) + ' / ' + str(img_len))
    if full_key_code == 32:
        cv2.imwrite('C:/Users/Taran/Desktop/' + 'Img' + str(n) + '.png', superimposed)
        print('image ' + str(current_img) + ' saved!')
        
