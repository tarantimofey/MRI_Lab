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
start_time = time.time()
print('Programm start, time:', start_time)



IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1


PATH = []

"""
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
"""
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




model_path = '/media/taran/SSD2/Data_Unet_from_laptop/Rat_Unet_article_2023/Trained_models/unet3plus_deepmeta/LUNG-PAT_multiclass/'
#model_path = 'C:/Users/Taran/Desktop/Data_Unet/Models/mct/edema/1Run/'
#model_file = model_path + 'MRI_lung'
model_file = model_path + 'LUNG'


image_folder = '/EXP_DATA/'
#mask_folder = '/Mask_edema/'
mask_folder = '/MASK_EXP/'
mask_pat_folder = '/MASK_PAT/'


augmetations_coeff = 1.1
n_of_epochs = 200

check_folders_flag = 0
test_split_coeff = 0.02

model_type = 0
batch_size = 32

save_test_images_flag = 1


model = tf.keras.models.load_model('/media/taran/SSD2/Data_Unet_from_laptop/Rat_Unet_article_2023/unet3plus_2class.h5')



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
    print('\nIoU_coeff = ', jaccard_score(Y_img, Y_img_ground, average='micro', zero_division=0))


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
    temp = listdir(path + mask_pat_folder)
    temp.sort(key=len)
    image_ids.append(temp)




X = []
Y = []
Y_pat = []

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

        mask_pat = imread(path + mask_pat_folder + id_)[:,:]
        #mask_pat = cv2.bitwise_or(mask_pat, mask_pat, mask_pat=img_mask_pat)
        mask_pat = cv2.normalize(mask_pat, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        mask_pat = mask_pat[..., np.newaxis]
        mask_pat = resize(mask_pat, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        Y_pat.append(mask_pat)




print('Done!\n')
X = np.asarray(X)
Y = np.asarray(Y)
Y_pat = np.asarray(Y_pat)
dataset_len = len(X)
print(str(dataset_len) + ' images were found\n')



if augmetations_coeff != 0:
    print('Augmenting data...')
    import scipy
    X_augmented = []
    Y_augmented = []
    Y_pat_augmented = []
    for i in tqdm(range(int(dataset_len*(augmetations_coeff-1)))):
        i = i%dataset_len
        X_img = X[i][:, :, 0]
        Y_img = Y[i][:, :, 0]
        Y_pat_img = Y[i][:, :, 0]
        im_merge = np.concatenate((X_img[..., None], Y_img[..., None], Y_pat_img[..., None]), axis=2)
        im_merge_t = elastic_transform(
            im_merge,
            im_merge.shape[1] * 0.5,
            im_merge.shape[1] * 0.09,
            im_merge.shape[1] * 0,
        )  # soft transform        
        
        test_img_t = im_merge_t[..., 0]
        test_mask_t = im_merge_t[..., 1]
        test_mask_pat_t = im_merge_t[..., 2]
        test_img_t, test_mask_t, test_mask_pat_t = rotate_and_scale([test_img_t, test_mask_t, test_mask_pat_t], 3, 0.1)
                
        test_mask_t[test_mask_t<0.5] = 0
        test_mask_t[test_mask_t>=0.5] = 1
        test_mask_pat_t[test_mask_pat_t<0.5] = 0
        test_mask_pat_t[test_mask_pat_t>=0.5] = 1
        X_augmented.append(test_img_t[..., np.newaxis])
        Y_augmented.append(test_mask_t[..., np.newaxis])
        Y_pat_augmented.append(test_mask_pat_t[..., np.newaxis])
    X = np.concatenate((X, X_augmented), axis=0)
    Y = np.concatenate((Y, Y_augmented), axis=0)
    Y_pat = np.concatenate((Y_pat, Y_pat_augmented), axis=0)
    print('Done!\n')
    print('n of augmented images:', len(X)-dataset_len, '; dataset_len =', len(X)/dataset_len, '\bx')


Y_comb = []
for i in range(len(X)):
    Y_img = Y[i][:, :, 0].astype(np.uint8)
    Y_pat_img = Y_pat[i][:, :, 0].astype(np.uint8)
    img_combined = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
    img_combined[Y_img==1] = 1
    img_combined[Y_pat_img==1] = 2
    Y_comb.append(img_combined[..., np.newaxis])




"""
cv2.namedWindow('img_combined', cv2.WINDOW_KEEPRATIO)
cv2.imshow("img_combined", Y_comb[50][:,:,0])
cv2.waitKey()
exit()
"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split_coeff, random_state=1, shuffle=True)
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
Y_comb_test = model.predict(X_test, verbose=1)
print('Done!\n\n')


cv2.namedWindow('Y_comb_test_img', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Y_comb_test_img", Y_comb_test[0][:,:,0].astype(np.uint8))
cv2.waitKey()
exit()