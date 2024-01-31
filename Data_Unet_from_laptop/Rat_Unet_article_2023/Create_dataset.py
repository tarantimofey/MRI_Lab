import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math 
#import My_Preful_lib.My_Preful_lib as mp
from os import listdir
from os.path import isfile, join
import cv2
import imageio

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

PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/CONTROL/1rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/CONTROL/2rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/CONTROL/3rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/CONTROL/10rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/CONTROL/11rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/CONTROL/12rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/CONTROL/15rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/CONTROL/19rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/CONTROL/20rat/Png')
PATH.append('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/CONTROL/21rat/Png')

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

save_path = '/media/taran/SSD2/Data_Unet/Rat/Rat_Dataset_DeepMeta/'

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
    temp = listdir(path + lung_mask_folder)
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
        
        if os.path.exists(path + pat_mask_folder + id_):
            pat_mask = imread(path + pat_mask_folder + id_)[:,:]
        else:
            pat_mask = np.zeros((128,128), np.uint8)
        #mask = cv2.bitwise_or(mask, mask, mask=img_mask)
        #pat_mask = cv2.normalize(pat_mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        #pat_mask = resize(pat_mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        pat_mask = cv2.resize(pat_mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
        pat_mask = cv2.normalize(pat_mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)        
        pat_mask = pat_mask[..., np.newaxis]
        Y.append(pat_mask)
            
        lung_mask = imread(path + lung_mask_folder + id_)[:,:]
        #mask = cv2.bitwise_or(mask, mask, mask=img_mask)
        #lung_mask = cv2.normalize(lung_mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        lung_mask = cv2.resize(lung_mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
        lung_mask = cv2.normalize(lung_mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
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

Y_comb = []
for i in range(len(X)):
    Y_img = Y[i][:, :, 0].astype(np.uint8)
    Y_lung_img = LUNG_MASKS[i][:, :, 0].astype(np.uint8)
    img_combined = np.zeros((128,128), np.uint8)
    img_combined[Y_lung_img==1] = 1
    img_combined[Y_img==1] = 2
    Y_comb.append(img_combined[..., np.newaxis])





X_train, X_test, Y_train, Y_test, LUNG_MASKS_train, LUNG_MASKS_test, Y_comb_train, Y_comb_test = train_test_split(X, Y, LUNG_MASKS, Y_comb, test_size=test_split_coeff, random_state=1, shuffle=1)









if not os.path.exists(save_path):
    os.makedirs(save_path+'X/')
    os.makedirs(save_path+'Y_comb/')
    os.makedirs(save_path+'Y_lung/')
    os.makedirs(save_path+'Y_pat/')
    os.makedirs(save_path+'Split/Train/X/')
    os.makedirs(save_path+'Split/Train/Y_comb/')
    os.makedirs(save_path+'Split/Train/Y_lung/')
    os.makedirs(save_path+'Split/Train/Y_pat/')
    os.makedirs(save_path+'Split/Test/X/')
    os.makedirs(save_path+'Split/Test/Y_comb/')
    os.makedirs(save_path+'Split/Test/Y_lung/')
    os.makedirs(save_path+'Split/Test/Y_pat/')

for i in range(len(X)):
    cv2.imwrite(save_path+'X/img_'+str(i)+'.tif', X[i][:, :, 0].astype(np.uint8))
    cv2.imwrite(save_path+'Y_comb/img_'+str(i)+'.tif', Y_comb[i][:, :, 0].astype(np.uint8))
    cv2.imwrite(save_path+'Y_lung/img_'+str(i)+'.tif', LUNG_MASKS[i][:, :, 0].astype(np.uint8))
    cv2.imwrite(save_path+'Y_pat/img_'+str(i)+'.tif', Y[i][:, :, 0].astype(np.uint8))


for i in range(len(X_train)):
    cv2.imwrite(save_path+'Split/Train/X/img_'+str(i)+'.tif', X_train[i][:, :, 0].astype(np.uint8))
    cv2.imwrite(save_path+'Split/Train/Y_comb/img_'+str(i)+'.tif', Y_comb_train[i][:, :, 0].astype(np.uint8))
    cv2.imwrite(save_path+'Split/Train/Y_lung/img_'+str(i)+'.tif', LUNG_MASKS_train[i][:, :, 0].astype(np.uint8))
    cv2.imwrite(save_path+'Split/Train/Y_pat/img_'+str(i)+'.tif', Y_train[i][:, :, 0].astype(np.uint8))


for i in range(len(X_test)):
    cv2.imwrite(save_path+'Split/Test/X/img_'+str(i)+'.tif', X_test[i][:, :, 0].astype(np.uint8))
    cv2.imwrite(save_path+'Split/Test/Y_comb/img_'+str(i)+'.tif', Y_comb_test[i][:, :, 0].astype(np.uint8))
    cv2.imwrite(save_path+'Split/Test/Y_lung/img_'+str(i)+'.tif', LUNG_MASKS_test[i][:, :, 0].astype(np.uint8))
    cv2.imwrite(save_path+'Split/Test/Y_pat/img_'+str(i)+'.tif', Y_test[i][:, :, 0].astype(np.uint8))



