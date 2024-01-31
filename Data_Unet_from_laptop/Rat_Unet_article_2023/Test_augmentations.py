import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math 
#import My_Preful_lib.My_Preful_lib as mp
from os import listdir
from os.path import isfile, join
import cv2
import imageio

#import tensorflow as tf
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



import scipy.ndimage.filters as filters
import scipy

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1



test_img = cv2.imread('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/Png/EXP_DATA/img_57.png', 2)
test_mask = cv2.imread('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/Png/MASK_EXP/img_57.png', 2)




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

    dx = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )

    return scipy.ndimage.interpolation.map_coordinates(
        image, indices, order=1, mode="reflect"
    ).reshape(shape)



def rotate_and_scale(image, angle, scale_factor, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    rand_angle = random_state.triangular(-angle, 0, angle)
    rand_scale = random_state.triangular(1-scale_factor, 1, 1+scale_factor)
    print('rand_angle =', rand_angle, '; rand_scale =', rand_scale)

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

def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))

#draw_grid(test_img, 20)
#draw_grid(test_mask, 20)


im_merge = np.concatenate((test_img[..., None], test_mask[..., None]), axis=2)

im_merge_t = elastic_transform(
    im_merge,
    im_merge.shape[1] * 0.6,
    im_merge.shape[1] * 0.09,
    im_merge.shape[1] * 0,
)  # soft transform



#im_merge_t = elastic_transform(im_merge, 100, 100, 0)  

test_img_t = im_merge_t[..., 0]
test_mask_t = im_merge_t[..., 1]
test_img_t, test_mask_t = rotate_and_scale([test_img_t, test_mask_t], 5, 0.2)



test_mask_t[test_mask_t>=127] = 255
test_mask_t[test_mask_t<127] = 0




cv2.namedWindow('test_img', cv2.WINDOW_KEEPRATIO)
cv2.imshow("test_img", test_img)
cv2.namedWindow('test_mask', cv2.WINDOW_KEEPRATIO)
cv2.imshow("test_mask", test_mask)
cv2.namedWindow('test_img_t', cv2.WINDOW_KEEPRATIO)
cv2.imshow("test_img_t", test_img_t)
cv2.namedWindow('test_mask_t', cv2.WINDOW_KEEPRATIO)
cv2.imshow("test_mask_t", test_mask_t)





cv2.moveWindow('test_img', 800, 200)
cv2.moveWindow('test_mask', 1200, 200)

cv2.moveWindow('test_img_t', 800, 600)
cv2.moveWindow('test_mask_t', 1200, 600)








cv2.waitKey()