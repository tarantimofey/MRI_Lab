#C:/Users/Taran/AppData/Local/Programs/Python/Python39/python.exe -i "$(FULL_CURRENT_PATH)"

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


IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 1


PATH = 'C:/Users/Taran/Desktop/Ready_data/OP_NS/Png/'
#PATH = 'C:/Users/Taran/Desktop/Data_Unet/View_test/'
#PATH = 'C:/Users/Taran/Desktop/Data_Unet/TRAIN/'
PATH = 'C:/Users/Taran/Desktop/mct/10/10_5/Png'

image_path = 'K:/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/Png/EXP_DATA/'
mask_path = 'K:/Data_Unet/Rat/Olya_26_07_23/Unet_masks_pat/BLEOMICINE/13rat/png/'


#image_folder = '/Image/'
#mask_folder = '/Mask_edema/'

def Show_images(n):

    X_img = X[n][:, :, 0].astype(np.uint8)
    Y_img = Y[n][:, :, 0].astype(np.uint8)

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


img_ids = [ f for f in listdir(image_path) if isfile(join(image_path,f)) ]
img_ids.sort(key=len) 
mask_ids = [ f for f in listdir(mask_path) if isfile(join(mask_path,f)) ]
mask_ids.sort(key=len) 


X = np.zeros((len(img_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

print('\nResizing training images and masks...')
for n in tqdm(range(0, len(mask_ids))):
    path = PATH
    img = imread(image_path + mask_ids[n])[:,:]
    img = img[..., np.newaxis]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #img = img.astype(float)/np.max(img)
    X[n] = img
        
for n in tqdm(range(0, len(mask_ids))):
    mask = imread(mask_path + mask_ids[n])[:,:]
    mask = mask[..., np.newaxis]
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    Y[n] = mask
print('Done!\n\n')

#X = X[..., np.newaxis]
#Y = Y[..., np.newaxis]


current_img = 0
img_len = len(X)
Show_images(current_img)
print('image ' + str(current_img) + ' / ' + str(img_len))

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