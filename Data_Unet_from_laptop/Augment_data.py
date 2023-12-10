import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math 
import My_Preful_lib.My_Preful_lib as mp
from os import listdir
from os.path import isfile, join
import cv2
import imageio

import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import random
from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize


from tensorflow.keras import layers

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1




PATH = []


PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/Png')






image_folder = '/EXP_DATA/'
#mask_folder = '/Mask_edema/'
mask_folder = '/MASK_EXP/'
#image_folder_2 = '/Image_insp/'
#mask_folder_2 = '/Mask_insp/'






def Show_images(n):

    X_img = X_train[n][:, :, 0].astype(np.uint8)
    Y_img = Y_train[n][:, :, 0].astype(np.uint8)

    #X_img = X_img*255
    #Y_img = Y_img*255
    X_img = cv2.normalize(X_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img = cv2.normalize(Y_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    cv2.namedWindow('X', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X", X_img)
    #cv2.resizeWindow('X', 200, 200)
    cv2.namedWindow('Y', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Y", Y_img)
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
    
    
    X_img_aug = X_train_aug[n][:, :].astype(np.uint8)
    Y_img_aug = Y_train_aug[n][:, :].astype(np.uint8)

    #X_img_aug = X_img_aug*255
    #Y_img_aug = Y_img_aug*255
    X_img_aug = cv2.normalize(X_img_aug, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img_aug = cv2.normalize(Y_img_aug, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    cv2.namedWindow('X_aug', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X_aug", X_img_aug)
    #cv2.resizeWindow('X', 200, 200)
    cv2.namedWindow('Y_aug', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Y_aug", Y_img_aug)
    #cv2.resizeWindow('Y', 200, 200)


    Y_img_aug_color = Y_img_aug.copy()
    Y_img_aug_color = cv2.cvtColor(Y_img_aug_color, cv2.COLOR_GRAY2RGB)

    for i in range(0, len(Y_img_aug_color)):
        for j in range(0, len(Y_img_aug_color)):
            Y_img_aug_color[i][j][0] = 0
            Y_img_aug_color[i][j][1] = 0       

    X_img_aug_color = X_img_aug.copy()
    X_img_aug_color = cv2.cvtColor(X_img_aug_color, cv2.COLOR_GRAY2RGB)
    global superimposed_aug
    superimposed_aug  = cv2.addWeighted(X_img_aug_color, 1, Y_img_aug_color, 0.5, 0)

    cv2.namedWindow('Img+mask_aug', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Img+mask_aug", superimposed_aug)


def augment(image, mask, seed):
    #image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    
    # Make a new seed.
    #new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    #seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    
    # Random crop back to the original size.
    #image = tf.image.stateless_random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
    
    # Random brightness.
    #image = tf.image.stateless_random_brightness(image, max_delta=0.08, seed=seed)
    
    #image = tf.clip_by_value(image, 0, 1)
    
    #image = tf.image.stateless_random_contrast(image, 0.7, 1.5, seed)
    #image = tf.image.stateless_random_jpeg_quality(image, 10, 100, seed)
    #image = tf.image.central_crop(image, 0.9)

    data_augmentation = tf.keras.Sequential([
    #layers.RandomFlip("horizontal_and_vertical"),
    #layers.RandomRotation(0.005),
    layers.RandomRotation(0.2),
    #layers.RandomZoom(0.2, 0.2, fill_mode='reflect', seed=seed),
    #layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
    ])
    #image = image[np.newaxis, ...]
    #image = data_augmentation(image)
    #mask = data_augmentation(mask)
    #mask = data_augmentation(mask)
    
    image = image[:, :, 0]
    mask = mask[:, :, 0]
    M = cv2.getRotationMatrix2D(center=(IMG_HEIGHT//2, IMG_WIDTH//2), angle=45, scale=1)
    image = cv2.warpAffine(image, M, (IMG_HEIGHT, IMG_WIDTH), borderMode=cv2.BORDER_REFLECT )
    mask  = cv2.warpAffine(mask, M, (IMG_HEIGHT, IMG_WIDTH))
    
    return image, mask
    
def process_img(img, mask, seed):
    img = tf.stack([img, mask])
    #img = tf.image.random_hue(img, max_delta=.5)
    #img = tf.image.central_crop(img, 0.9)
    img = tf.image.stateless_random_brightness(img, max_delta=40, seed=seed)
    #img = tf.image.stateless_random_brightness(img, max_delta=100, seed=seed)
    img = tf.image.stateless_random_contrast(img, 0.7, 1.5, seed)
    #img = tf.image.stateless_random_crop(img, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
    return img[0], img[1]



image_ids = []
for i, path in enumerate(PATH):
    temp = listdir(path + mask_folder)
    temp.sort(key=len)
    image_ids.append(temp)

X_train = []
Y_train = []

print('\nResizing training images...')
for i, path in enumerate(PATH):
    print('Dataset ' + str(i) + ':')
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
        X_train.append(img)
            
        mask = imread(path + mask_folder + id_)[:,:]
        #mask = cv2.bitwise_or(mask, mask, mask=img_mask)
        mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        mask = mask[..., np.newaxis]
        mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        Y_train.append(mask)

print('Done!\n')
X_train = np.asarray(X_train)
Y_train= np.asarray(Y_train)
print(str(len(X_train)) + ' images were found/n/n')

#X_train = X_train[50:100]
#Y_train = Y_train[50:100]

print(X_train.shape)




# Create a generator.
rng = tf.random.Generator.from_seed(1234, alg='philox')
#seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

X_train_aug = []
Y_train_aug = []

seed = rng.make_seeds(2)[0]
for i in range(len(X_train)):
    #X_train_img, Y_train_img = augment(X_train[i], Y_train[i], seed)
    X_train_img, Y_train_img = process_img(X_train[i], Y_train[i], seed)
    
    #X_train_img = X_train[i][:, :, 0].astype(np.uint8)
    #Y_train_img = Y_train[i][:, :, 0].astype(np.uint8)
    #X_train_img = X_train[i][:, :, 0]
    #Y_train_img = Y_train[i][:, :, 0]

    #X_img = X_img*255
    #Y_img = Y_img*255
    #X_train_img = cv2.normalize(X_train_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #Y_train_img = cv2.normalize(Y_train_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #Y_train_img = Y_train_img[np.newaxis, :, :, np.newaxis]
    #X_train_img, Y_train_img = seq(image=X_train_img, segmentation_maps=Y_train_img)
    #Y_train_img = Y_train_img[0, :, :, 0]
    
    #X_train_img = cv2.resize(X_train_img, (152, 152), interpolation=cv2.INTER_NEAREST)
    X_train_aug.append(X_train_img)
    Y_train_aug.append(Y_train_img)


X_train_aug = np.asarray(X_train_aug)
Y_train_aug = np.asarray(Y_train_aug)

current_img = 0
Show_images(current_img)

img_len = len(X_train)
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
        
