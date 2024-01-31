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
from sklearn.metrics import jaccard_score
import os
import random
from tqdm import tqdm

import pltable
from pltable import PrettyTable

from skimage.io import imread, imshow
from skimage.transform import resize
import time

import keras_unet_collection
from keras_unet_collection import models, base, utils

start_time = time.time()
print('Programm start, time:', start_time)



IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1


PATH = []

PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/22rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/23rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/24rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/26rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/27rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/28rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/29rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/30rat/Png')

PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/TREATMENT/4rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/TREATMENT/5rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/TREATMENT/7rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/TREATMENT/8rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/TREATMENT/9rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/TREATMENT/14rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/TREATMENT/16rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/TREATMENT/17rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/TREATMENT/18rat/Png')









model_path = '/media/taran/SSD2/Data_Unet_from_laptop/Rat_Unet_article_2023/Trained_models/unet3plus/PAT_affine-2Run/'
#model_path = 'C:/Users/Taran/Desktop/Data_Unet/Models/mct/edema/1Run/'
#model_file = model_path + 'MRI_lung'
model_file = model_path + 'Pat'


image_folder = '/EXP_DATA/'
pat_mask_folder = '/MASK_PAT/'
lung_mask_folder = '/MASK_EXP/'


augmetations_coeff = 3
n_of_epochs = 200


check_folders_flag = 0 #SystemExit
test_split_coeff = 0.02

model_type = 0
batch_size = 32



save_test_images_flag = 1



if not os.path.exists(model_path):
    os.makedirs(model_path)
    

if model_type == 0:
    #model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/MRI_lung_0.h5')
    #model = tf.keras.models.load_model('/media/taran/SSD2/Data_Unet_from_laptop/Rat_Unet_article_2023/unet3plus.h5')
    model = keras_unet_collection.models.unet_3plus_2d((128,128,1),1,[64,128,256,512,1024], weights=None)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + random_state.uniform(
        -alpha_affine, alpha_affine, size=pts1.shape
    ).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
    )

    dx = scipy.ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = scipy.ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )

    return scipy.ndimage.map_coordinates(
        image, indices, order=1, mode="reflect"
    ).reshape(shape)



def rotate_and_scale(image, angle, scale_factor, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    rand_angle = random_state.triangular(-angle, 0, angle)
    rand_scale = random_state.triangular(1-scale_factor, 1, 1+scale_factor)
    #print('rand_angle =', rand_angle, '; rand_scale =', rand_scale)

    # get image height, width
    (h, w) = image[0].shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, rand_angle, rand_scale)
    out = []
    for img in image:
        rotated = cv2.warpAffine(img, M, (h, w), borderMode=cv2.BORDER_REFLECT_101)
        out.append(rotated)
    return out


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
    Y_img_ground = Y_ground[n][:, :, 0].astype(np.uint8)
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
    temp = listdir(path + pat_mask_folder)
    temp.sort(key=len)
    image_ids.append(temp)

X = []
Y = []
LUNG_MASKS = []

print('\nResizing training images...')
for i, path in enumerate(PATH):
    print('Dataset ' + str(i) + ':')
    print(PATH[i] + '   ' + image_folder + '<--->' + pat_mask_folder + '<--->' + lung_mask_folder)
    for n, id_ in tqdm(enumerate(image_ids[i]), total=len(image_ids[i])):
        #path = PATH
        img = imread(path + image_folder + id_)[:,:]
        #image_mask_folder = '/Mask/'
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
            
        pat_mask = imread(path + pat_mask_folder + id_)[:,:]
        #mask = cv2.bitwise_or(mask, mask, mask=img_mask)
        #pat_mask = cv2.normalize(pat_mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        #pat_mask = resize(pat_mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        pat_mask = cv2.resize(pat_mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
        pat_mask = pat_mask[..., np.newaxis]
        Y.append(pat_mask)
            
        lung_mask = imread(path + lung_mask_folder + id_)[:,:]
        #mask = cv2.bitwise_or(mask, mask, mask=img_mask)
        lung_mask = cv2.normalize(lung_mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        lung_mask = cv2.resize(lung_mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
        lung_mask = cv2.normalize(lung_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        lung_mask = lung_mask[..., np.newaxis]
        LUNG_MASKS.append(lung_mask)


print('Done!\n')
X = np.asarray(X)
Y = np.asarray(Y)
LUNG_MASKS = np.asarray(LUNG_MASKS)
print(X.shape, Y.shape, LUNG_MASKS.shape)
dataset_len = len(X)
print(str(dataset_len) + ' images were found\n')

if check_folders_flag == 1:
    raise SystemExit

if augmetations_coeff != 0:
    print('Augmenting data...')
    import scipy
    X_augmented = []
    Y_augmented = []
    LUNG_MASKS_augmented = []
    for i in tqdm(range(int(dataset_len*(augmetations_coeff-1)))):
        i = i%dataset_len
        X_img = X[i][:, :, 0]
        Y_img = Y[i][:, :, 0]
        LUNG_MASK_img = LUNG_MASKS[i][:, :, 0]
        im_merge = np.concatenate((X_img[..., None], Y_img[..., None], LUNG_MASK_img[..., None]), axis=2)
        im_merge_t = elastic_transform(
            im_merge,
            im_merge.shape[1] * 0.5,
            im_merge.shape[1] * 0.09,
            im_merge.shape[1] * 0,
        )  # soft transform        
        
        test_img_t = im_merge_t[..., 0]
        test_mask_t = im_merge_t[..., 1]
        test_lung_mask_t = im_merge_t[..., 2]
        test_img_t, test_mask_t, test_lung_mask_t = rotate_and_scale([test_img_t, test_mask_t, test_lung_mask_t], 3, 0.1)
                
        test_mask_t[test_mask_t<0.5] = 0
        test_mask_t[test_mask_t>=0.5] = 1
        test_lung_mask_t[test_lung_mask_t<0.5] = 0
        test_lung_mask_t[test_lung_mask_t>=0.5] = 1
        X_augmented.append(test_img_t[..., np.newaxis])
        Y_augmented.append(test_mask_t[..., np.newaxis])
        LUNG_MASKS_augmented.append(test_lung_mask_t[..., np.newaxis])
    X = np.concatenate((X, X_augmented), axis=0)
    Y = np.concatenate((Y, Y_augmented), axis=0)
    LUNG_MASKS = np.concatenate((LUNG_MASKS, LUNG_MASKS_augmented), axis=0)
    LUNG_MASKS = LUNG_MASKS.astype(np.uint8)
    print('Done!\n')
    print('n of augmented images:', len(X)-dataset_len, '; dataset_len =', len(X)/dataset_len, '\bx')


X_ground = X.copy()









Y_th = []
coord_for_AT = []
for i in tqdm(range(len(X))):    
    cnts = cv2.findContours(LUNG_MASKS[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    img = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
    bounding_boxes_x1 = []
    bounding_boxes_y1 = []
    bounding_boxes_x2 = []
    bounding_boxes_y2 = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        bounding_boxes_x1.append(x)
        bounding_boxes_y1.append(y)
        bounding_boxes_x2.append(x+w)
        bounding_boxes_y2.append(y+h)
    bounding_boxes_x1 = sorted(bounding_boxes_x1, reverse=False)
    bounding_boxes_y1 = sorted(bounding_boxes_y1, reverse=False)
    bounding_boxes_x2 = sorted(bounding_boxes_x2, reverse=True)
    bounding_boxes_y2 = sorted(bounding_boxes_y2, reverse=True)
    if len(bounding_boxes_x1) != 0 and len(bounding_boxes_x2) != 0 and len(bounding_boxes_y1) != 0 and len(bounding_boxes_y2) != 0:
        img = cv2.rectangle(img, (bounding_boxes_x1[0], bounding_boxes_y1[0]), (bounding_boxes_x2[0], bounding_boxes_y2[0]), 255, -1)
        coord_for_AT.append([(bounding_boxes_x1[0], bounding_boxes_y1[0]), (bounding_boxes_x2[0], bounding_boxes_y2[0]), (bounding_boxes_x1[0], bounding_boxes_y2[0])])
    else:
        coord_for_AT.append([(0,0), (128,128), (0, 128)])
    Y_th.append(img)


    
X_affine = []
Y_affine = []
print('\nAffine transforming masks...')
for i in tqdm(range(len(X))):
    X_img_affine = X[i]
    Y_img_affine = Y[i]
    pts1 = np.float32(coord_for_AT[i])
    pts2 = np.float32([(0,0), (128,128), (0, 128)])
    matrix = cv2.getAffineTransform(pts1, pts2)
    X_img_affine = cv2.warpAffine(X_img_affine, matrix, (len(img), len(img)))
    #Y_img_affine = cv2.warpAffine(Y_img_affine, matrix, (len(img), len(img)))
    Y_img_affine = cv2.warpAffine(Y_img_affine, matrix, (len(img), len(img)), cv2.INTER_NEAREST)
    Y_img_affine = (Y_img_affine > 0.5).astype(np.uint8)
    Y_img_affine*=1
    X_img_affine = X_img_affine[..., np.newaxis]
    Y_img_affine = Y_img_affine[..., np.newaxis]
    X_affine.append(X_img_affine)
    Y_affine.append(Y_img_affine)
    
X_affine = np.asarray(X_affine)
Y_affine = np.asarray(Y_affine)
    



X_indeces = np.arange(len(X_affine))
X_ground_train, X_ground_test, X_train, X_test, Y_train, Y_test, coord_for_AT_train, coord_for_AT_test, LUNG_MASKS_train, LUNG_MASKS_test = train_test_split(X_ground, X_affine, Y_affine, coord_for_AT, LUNG_MASKS, test_size=test_split_coeff, random_state=1, shuffle=1)
print(str(len(X_train)) + ' images selected for training (' + str(test_split_coeff*100) + '%)\n\n')
Y_ground = Y_test.copy()
checkpointer = tf.keras.callbacks.ModelCheckpoint(model_file+'.h5', verbose=1, save_best_only='True')
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir=model_path)
    #tf.keras.callbacks.LearningRateScheduler(scheduler)
]


train_begin_time = time.time()

print('Fitting model...')
#with tf.device('/GPU:1'):
results = model.fit(X_train,Y_train, validation_split=0.1, batch_size=batch_size, epochs=n_of_epochs, callbacks=[callbacks, checkpointer])
print('Done!\n')

train_end_time = time.time()

print('Fitting test data...')
Y_test = model.predict(X_test, verbose=1)
Y_test = (Y_test > 0.5).astype(np.uint32)
Y_test *= 255
print('Done!\n\n')


X_affine_reversed = []
Y_affine_reversed = []
Y_ground_reversed = []
print('Affine transforming masks...')
for i in tqdm(range(len(X_test))):
    X_img_affine = X_test[i][:,:,0].astype(np.uint8)
    Y_img_affine = Y_test[i][:,:,0].astype(np.uint8)
    Y_ground_img_affine = Y_ground[i][:,:,0].astype(np.uint8)
    pts1 = np.float32(coord_for_AT_test[i])
    pts2 = np.float32([(0,0), (128,128), (0, 128)])
    matrix = cv2.getAffineTransform(pts2, pts1)
    X_img_affine_reversed = cv2.warpAffine(X_img_affine, matrix, (len(img), len(img)))
    #Y_img_affine = cv2.warpAffine(Y_img_affine, matrix, (len(img), len(img)))
    Y_img_affine_reversed = cv2.warpAffine(Y_img_affine, matrix, (len(img), len(img)), cv2.INTER_NEAREST)
    Y_ground_img_affine_reversed = cv2.warpAffine(Y_ground_img_affine, matrix, (len(img), len(img)), cv2.INTER_NEAREST)
    Y_img_affine_reversed = (Y_img_affine_reversed > 0.5).astype(np.uint8)
    Y_img_affine_reversed*=1
    Y_ground_img_affine_reversed = (Y_ground_img_affine_reversed > 0.5).astype(np.uint8)
    Y_ground_img_affine_reversed*=1
    X_img_affine_reversed = X_img_affine_reversed[..., np.newaxis]
    Y_img_affine_reversed = Y_img_affine_reversed[..., np.newaxis]
    Y_ground_img_affine_reversed = Y_ground_img_affine_reversed[..., np.newaxis]
    X_affine_reversed.append(X_img_affine_reversed[..., np.newaxis])
    Y_affine_reversed.append(Y_img_affine_reversed[..., np.newaxis])
    Y_ground_reversed.append(Y_ground_img_affine_reversed[..., np.newaxis])
    
X_affine_reversed = np.asarray(X_affine_reversed)
Y_affine_reversed = np.asarray(Y_affine_reversed)
Y_ground_reversed = np.asarray(Y_ground_reversed)


X_test = X_ground_test
Y_test = Y_affine_reversed
Y_ground = Y_ground_reversed







dice_table = []
for i in range(len(Y_test)):
    Y_img = Y_test[i][:, :, 0].astype(np.uint8)
    Y_img_ground = Y_ground[i][:, :, 0].astype(np.uint8)
    Y_img = cv2.normalize(Y_img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img_ground = cv2.normalize(Y_img_ground, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    d = Dice_coeff(Y_img, Y_img_ground)
    dice_table.append(d)
dice_table = np.asarray(dice_table)
print('Dice_coefficient:\nmean = ', dice_table.mean(), '\nmax = ', dice_table.max(), '\nmin = ', dice_table.min(), '\n\n')

IoU_table = []
for i in range(len(Y_test)):
    Y_img = Y_test[i][:, :, 0].astype(np.uint8)
    Y_img_ground = Y_ground[i][:, :, 0].astype(np.uint8)
    Y_img = cv2.normalize(Y_img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img_ground = cv2.normalize(Y_img_ground, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    IoU = jaccard_score(Y_img_ground, Y_img, average='micro')
    IoU_table.append(IoU)
IoU_table = np.asarray(IoU_table)
print('IoU_coefficient:\nmean = ', IoU_table.mean(), '\nmax = ', IoU_table.max(), '\nmin = ', IoU_table.min(), '\n\n')


if save_test_images_flag == 1:
    if not os.path.exists(model_path+'/test_images'):
        os.makedirs(model_path+'/test_images/X')
        os.makedirs(model_path+'/test_images/Y')
        os.makedirs(model_path+'/test_images/Y_ground')
    for i in range(len(Y_test)):
        cv2.imwrite(model_path+'/test_images/X/' + 'test_img_' + str(i) + '.png', cv2.normalize(X_test[i][:,:,0].astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        cv2.imwrite(model_path+'/test_images/Y/' + 'test_img_' + str(i) + '.png', cv2.normalize(Y_test[i][:,:,0].astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        cv2.imwrite(model_path+'/test_images/Y_ground/' + 'test_img_' + str(i) + '.png', cv2.normalize(Y_ground[i][:,:,0].astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))



end_time = time.time()
execution_time = end_time - start_time
train_time = train_end_time - train_begin_time
print('Execution time:', execution_time, 's = ', execution_time/60, 'min')
print('Train time:', train_time, 's = ', train_time/60, 'min')


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
    Path_table.add_row([1, image_folder + '<--->' + pat_mask_folder, str(len(image_ids[i]))])
    #txt_file.writelines('\nfolder_1: ' + image_folder + '<--->' + mask_folder + '; N_images: ' + str(len(image_ids[i])))
    #txt_file.writelines(Path_table.get_string())
    txt_file.writelines(str(Path_table))
    txt_file.writelines('\n\n')
if augmetations_coeff == 0:
    txt_file.writelines('No augmentations were performed')
else:
    txt_file.writelines('N of images in dataset: ' + str(dataset_len) + '\n')    
    txt_file.writelines('N of augmented images: ' + str(len(X)-dataset_len) + '; dataset_len = ' + str(round(len(X)/dataset_len, 4)) + 'x\n')
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
txt_file.writelines('\nIoU_coefficient:\nmean = ' + str(IoU_table.mean()) + '\nmax = ' + str(IoU_table.max()) + '\nmin = ' + str(IoU_table.min())) 
txt_file.writelines('\n\n-----------------------------------------------------------------')   
txt_file.writelines('\nExecution time: ' + str(round((execution_time), 2)) + 's = ' + str(round((execution_time/60), 2)) + 'min')  
txt_file.writelines('\nTrain time: ' + str(round((train_time), 2)) + 's = ' + str(round((train_time/60), 2)) + 'min')  
txt_file.writelines('\n\n-----------------------------------------------------------------')
txt_file.close()

with open(model_file+'_summary.txt', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
with open(model_file+'_dice.txt', 'w') as f:
    for i in range(len(dice_table)):
        f.writelines(str(dice_table[i]) + ', ')
with open(model_file+'_IoU.txt', 'w') as f:
    for i in range(len(IoU_table)):
        f.writelines(str(IoU_table[i]) + ', ')

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


cv2.moveWindow('X', 1, 0)
cv2.moveWindow('Y', 400, 0)
cv2.moveWindow('Y_ground', 800, 0)

cv2.moveWindow('Img+mask', 400, 400)

cv2.moveWindow('Img+mask_ground', 800, 400)



img_len = len(X_test)
while(1):
    full_key_code = cv2.waitKeyEx(0)
    if full_key_code == 65363:
        current_img += 1
        if current_img >= img_len:
            current_img = img_len-1
        Show_images(current_img)
        print('image ' + str(current_img) + ' / ' + str(img_len))
    if full_key_code == 65361:
        current_img -= 1        
        if current_img < 0:
            current_img = 0
        Show_images(current_img)
        print('image ' + str(current_img) + ' / ' + str(img_len))
    if full_key_code == 27:
            exit()
        
