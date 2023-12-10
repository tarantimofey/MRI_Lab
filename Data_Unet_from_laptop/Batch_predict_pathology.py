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
import os
import random
from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize

import PIL



IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1


PATH = []


PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/1rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/2rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/3rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/10rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/11rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/12rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/15rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/19rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/20rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/21rat/png')

PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/22rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/23rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/24rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/26rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/27rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/28rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/29rat/png')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/30rat/png')
                                                            
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/4rat/png/')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/5rat/png/')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/7rat/png/')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/8rat/png/')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/9rat/png/')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/14rat/png/')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/16rat/png/')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/17rat/png/')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/18rat/png/')


image_folder = '/EXP_DATA/'
pat_mask_folder = '/MASK_PAT/'
lung_mask_folder = '/MASK_EXP/'


model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/Olya_26_07_23/PAT-2Run/Pat.h5')

Common_path = os.path.commonpath(PATH)
Common_path = 'K:\Data_Unet/Rat/Olya_26_07_23/'
Save_path = Common_path + '/Unet_masks_pat/'
if not os.path.exists(Save_path):
    os.mkdir(Save_path)
    
#path1 = os.path.splitdrive()
#print(PATH[0][len(Common_path):])

print('\n')

#raise SystemExit





for i in range(len(PATH)):
    print('\n\n' + '#'*os.get_terminal_size()[0])
    test_ids = []
    temp = listdir(PATH[i] + image_folder)
    temp.sort(key=len)
    test_ids = temp
    
    X = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    LUNG_MASKS = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    print('\nResizing training images and masks in dataset ' + str(i) + '...')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        img = imread(PATH[i] + image_folder + id_)[:,:]
        img = img[..., np.newaxis]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X[n] = img
        #######################
        
        if os.path.exists(PATH[i] + lung_mask_folder + id_):
            lung_mask = imread(PATH[i] + lung_mask_folder + id_)[:,:]
        else:
            lung_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        #mask = cv2.bitwise_or(mask, mask, mask=img_mask)
        lung_mask = cv2.normalize(lung_mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        #lung_mask = lung_mask[..., np.newaxis]        
        lung_mask = cv2.resize(lung_mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
        lung_mask = cv2.normalize(lung_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        LUNG_MASKS[n] = lung_mask
        
        
    Y_th = []
    coord_for_AT = []
    for n in tqdm(range(len(X))):    
        cnts = cv2.findContours(LUNG_MASKS[n], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    print('Affine transforming masks...')
    for n in tqdm(range(len(X))):
        X_img_affine = X[n]
        #Y_img_affine = Y[n]
        pts1 = np.float32(coord_for_AT[n])
        pts2 = np.float32([(0,0), (128,128), (0, 128)])
        matrix = cv2.getAffineTransform(pts1, pts2)
        X_img_affine = cv2.warpAffine(X_img_affine, matrix, (len(img), len(img)))
        #Y_img_affine = cv2.warpAffine(Y_img_affine, matrix, (len(img), len(img)))
        #Y_img_affine = cv2.warpAffine(Y_img_affine, matrix, (len(img), len(img)), cv2.INTER_NEAREST)
        #Y_img_affine = (Y_img_affine > 0.5).astype(np.uint8)
        #Y_img_affine*=1
        #X_img_affine = X_img_affine[..., np.newaxis]
        #Y_img_affine = Y_img_affine[..., np.newaxis]
        X_affine.append(X_img_affine)
        #Y_affine.append(Y_img_affine)
        
    X_affine = np.asarray(X_affine)
    #Y_affine = np.asarray(Y_affine)
        
        
        
        
    Y_antiaffine = []
    print('\nPredicting dataset ' + str(i) + '...')
    Y = model.predict(X_affine, verbose=1)
    Y = (Y > 0.5).astype(np.uint8)
    Y *= 255
    print('Done!')
    
    print('Affine antitransforming masks...')
    for n in tqdm(range(len(Y))):
        Y_img_antiaffine = Y[n]
        #Y_img_affine = Y[n]
        pts1 = np.float32([(0,0), (128,128), (0, 128)])
        pts2 = np.float32(coord_for_AT[n])
        matrix = cv2.getAffineTransform(pts1, pts2)
        Y_img_antiaffine = cv2.warpAffine(Y_img_antiaffine, matrix, (len(img), len(img)))
        #Y_img_affine = cv2.warpAffine(Y_img_affine, matrix, (len(img), len(img)))
        #Y_img_affine = cv2.warpAffine(Y_img_affine, matrix, (len(img), len(img)), cv2.INTER_NEAREST)
        #Y_img_affine = (Y_img_affine > 0.5).astype(np.uint8)
        #Y_img_affine*=1
        #X_img_affine = X_img_affine[..., np.newaxis]
        #Y_img_affine = Y_img_affine[..., np.newaxis]
        Y_antiaffine.append(Y_img_antiaffine)
        #Y_affine.append(Y_img_affine)
        
    Y_antiaffine = np.asarray(Y_antiaffine)
    #Y_affine = np.asarray(Y_affine)
    
    
    
    print('\nSavings masks for dataset ' + str(i) + '...')
    Folder_path = Save_path + PATH[i][len(Common_path):]
    if not os.path.exists(Folder_path):
        os.makedirs(Folder_path)
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        img = Y_antiaffine[n][:,:]
        #img = resize(img, (152, 152), mode='constant', preserve_range=True)
        #img = img[:,:,0]
        img = cv2.resize(img, (152, 152), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(Folder_path + '/' + id_, img)
    print('Done!')

"""
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.namedWindow('Img+mask_ground', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Img+mask_ground", superimposed_ground)
"""