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



IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

path = '/media/taran/SSD2/Data_Unet_from_laptop/Rat_Unet_article_2023/Trained_models/unet3plus_deepmeta/PAT_affine-2Run/test_images'
Y_folder = path + '/Y_deepmeta/pat/'
Y_ground_folder = path + '/Y_ground/'




def Dice_coeff(gt, seg):
    k = seg.max()
    dice = np.sum(seg[gt==k]==k)*2.0 / (np.sum(seg[seg==k]==k) + np.sum(gt[gt==k]==k))
    return dice











Y_ids = listdir(Y_folder)
Y_ids.sort(key=len)


Y_ground_ids = listdir(Y_ground_folder)
Y_ground_ids.sort(key=len)


Y = []
Y_ground = []
for n, id_ in tqdm(enumerate(Y_ids), total=len(Y_ids)):
    Y_img = imread(Y_folder + id_)[:,:]
    #mask = cv2.bitwise_or(mask, mask, mask=img_mask)
    Y_img = cv2.normalize(Y_img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img = cv2.normalize(Y_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y.append(Y_img)

for n, id_ in tqdm(enumerate(Y_ground_ids), total=len(Y_ground_ids)):
    Y_ground_img = imread(Y_ground_folder + id_)[:,:]
    #mask = cv2.bitwise_or(mask, mask, mask=img_mask)
    Y_ground_img = cv2.normalize(Y_ground_img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_ground_img = cv2.normalize(Y_ground_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_ground.append(Y_ground_img)

dice_table = []
IoU_table = []
for i in range(len(Y)):
    Y_img = Y[i]
    Y_img_ground = Y_ground[i]
    d = Dice_coeff(Y_img, Y_img_ground)
    IoU = jaccard_score(Y_img_ground, Y_img, average='micro')
    dice_table.append(d)
    IoU_table.append(IoU)


dice_table = np.asarray(dice_table)
IoU_table = np.asarray(IoU_table)

print(dice_table.mean(), IoU_table.mean())



txt_file = open(path+'/metrics.txt', 'w', encoding="utf-8")
txt_file.writelines('-----------------------------------------------------------------\n') 
txt_file.writelines('N of images: ' + str(len(Y)) + '\n')
txt_file.writelines('\nDice_coefficient:\nmean = ' + str(dice_table.mean()) + '\nmax = ' + str(dice_table.max()) + '\nmin = ' + str(dice_table.min())) 
txt_file.writelines('\n\nIoU_coefficient:\nmean = ' + str(IoU_table.mean()) + '\nmax = ' + str(IoU_table.max()) + '\nmin = ' + str(IoU_table.min())) 
txt_file.writelines('\n\n-----------------------------------------------------------------')   
txt_file.writelines('\ndice_table = [')
for i in range(len(dice_table)):
    txt_file.writelines(str(dice_table[i]) + ', ')
txt_file.writelines('\b\b]')
txt_file.writelines('\n\nIoU_table = [')
for i in range(len(IoU_table)):
    txt_file.writelines(str(IoU_table[i]) + ', ')
txt_file.writelines('\b\b]')
txt_file.close()

