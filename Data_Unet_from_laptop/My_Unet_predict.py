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


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1


#path = 'C:/Users/Taran/Desktop/Data_Unet/TEST/Image/'
path = 'C:/Users/Taran/Desktop/Ready_data/OP_SS/Png/Image/'
#path = 'C:/Users/Taran/Desktop/Ready_data/OP_NS/Png/Image/'
#path = 'C:/Users/Taran/Desktop/Ready_data/OP_15/Png/Image/'
#path = 'C:/Users/Taran/Desktop/anonimus/Png/Image/'

# path = 'K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/21rat/Png/EXP_DATA/'
# path_ground = 'K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/21rat/Png/MASK_EXP/'
# path = 'K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/Png/EXP_DATA/'
# path_ground = 'K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/Png/MASK_EXP/'
path = 'K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/EXP_DATA/'
path_ground = 'K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/MASK_EXP/'

#path = 'C:/Users/Taran/Desktop/Data_Unet/Data/Rat/OE_LUNGS/LUNGS.hT1/15/Png/Insp/'

#model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/MRI_lung_0.h5')
#model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/Lung/4Run/MRI_lung_1.h5')
#model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/mct/6Run/MRI_lung.h5')
model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/Olya_26_07_23/3Run/MRI_lung.h5')


def Show_images(n):

    global X_img
    global Y_img
    global Y_img_refined
    global Y_img_ground
    global X_masked
    global X_affine
    global X_thresholded
    
    X_img = X_test[n][:, :, 0].astype(np.uint8)
    Y_img = Y_test[n][:, :, 0].astype(np.uint8)
    Y_img_refined = Y_refined[n][:, :].astype(np.uint8)
    Y_img_ground = Y_ground[n]
    X_img_masked = X_masked[n][:, :].astype(np.uint8)
    X_img_affine = X_affine[n][:, :].astype(np.uint8)
    X_img_thresholded = X_thresholded[n]
        
    X_img = cv2.normalize(X_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img = cv2.normalize(Y_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img_ground = cv2.normalize(Y_img_ground, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img_ground *= 255
    #Y_img_refined = cv2.normalize(Y_img_refined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    X_img_masked = cv2.normalize(X_img_masked, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    X_img_affine = cv2.normalize(X_img_affine, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    X_img_thresholded = cv2.normalize(X_img_thresholded, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    cv2.namedWindow('X', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X", X_img)
    #cv2.resizeWindow('X', 200, 200)
    cv2.namedWindow('Y', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Y", Y_img)
    #cv2.resizeWindow('Y', 200, 200)
    cv2.namedWindow('Y_refined', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Y_refined", Y_img_refined)
    #cv2.resizeWindow('Y', 200, 200)
    cv2.namedWindow('Y_ground', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Y_ground", Y_img_ground)
    #cv2.resizeWindow('Y', 200, 200)
    cv2.namedWindow('X_masked', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X_masked", X_img_masked)
    #cv2.resizeWindow('Y', 200, 200)
    cv2.namedWindow('X_affine', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X_affine", X_img_affine)
    #cv2.resizeWindow('Y', 200, 200)
    cv2.namedWindow('X_thresholded', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X_thresholded", X_img_thresholded)
    #cv2.resizeWindow('Y', 200, 200)


    global superimposed
    superimposed  = Superimpose(X_img, Y_img)
    cv2.namedWindow('Img+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Img+mask", superimposed)

    global superimposed_refined
    superimposed_refined  = Superimpose(X_img, Y_img_refined)
    cv2.namedWindow('Img+mask_refined', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Img+mask_refined", superimposed_refined)

    global superimposed_ground
    superimposed_ground  = Superimpose(X_img, Y_img_ground)
    cv2.namedWindow('Img+mask_ground', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Img+mask_ground", superimposed_ground)
    
    print('\nDice_coeff = ', Dice_coeff(Y_img, Y_img_ground))

def Save_images(n):
    cv2.imwrite('K:/' + 'Img' + str(n) + '_s.png', superimposed)
    cv2.imwrite('K:/' + 'Img' + str(n) + '_X.png', X_img)
    cv2.imwrite('K:/' + 'Img' + str(n) + '_Y.png', Y_img)
    cv2.imwrite('K:/' + 'Img' + str(n) + '_Y_refined.png', Y_img_refined)
    cv2.imwrite('K:/' + 'Img' + str(n) + '_Y_ground.png', Y_img_ground)
    cv2.imwrite('K:/' + 'Img' + str(n) + '_superimposed_refined.png', superimposed_refined)
    cv2.imwrite('K:/' + 'Img' + str(n) + '_superimposed_ground.png', superimposed_ground)
    print('image ' + str(current_img) + ' saved!')    
    
def Refine_mask(th):
    th_c = th.copy()
    th_c = cv2.medianBlur(th_c, 3)
    """
    for i in range(0, len(th_c)):
        clear=255
        cv2.floodFill(th_c, None, (i, 0), 255)
        cv2.floodFill(th_c, None, (i, len(th_c)-1), 255)
        cv2.floodFill(th_c, None, (0, i), 255)
        cv2.floodFill(th_c, None, (len(th_c)-1, i), 255)
    """
    cnt, hierarchy = cv2.findContours(th_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)
    th = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    if len(cnt) >= 2:
        cv2.drawContours(th, [cnt[0], cnt[1]], -1, 255, -1)
    if len(cnt) >= 1:
        cv2.drawContours(th, [cnt[0]], -1, 255, -1)
    if len(cnt) >= 3:
        cv2.drawContours(th, [cnt[0], cnt[1], cnt[2]], -1, 255, -1)
    #th_c = cv2.medianBlur(th_c, 3)
    return th

def Segment_light_areas(img):
    threshold_value = (img.max - img.min)/2
    th_light = cv2.threshold(th,threshold_value,255,cv.THRESH_BINARY)
    return th_light

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


test_ids = []
temp = listdir(path)
temp.sort(key=len)
test_ids = temp

#test_ids = [ f for f in listdir(path) if isfile(join(path,f)) ]
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint32)
print('\nResizing training images and masks...')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = imread(path + id_)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img[..., np.newaxis]
    X_test[n] = img
print('Done!\n')

Y_ground = []
print('\nResizing training images and masks...')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    if os.path.exists(path_ground + id_):
        img = imread(path_ground + id_)[:,:]
        #img = img[..., np.newaxis]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        #print(img.dtype)
        Y_ground.append(img)
    else:
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        Y_ground.append(img)
        
print('Done!\n')


print('Predicting...')
Y_test = model.predict(X_test, verbose=1)
Y_test = (Y_test > 0.5).astype(np.uint8)
#Y_test *= 255
print('Done!')

Y_refined = []
print('\nRefining masks...')
for i in tqdm(range(len(Y_test))):
    #th = Refine_mask(Y_test[i])
    th = Y_test[i]
    th = Refine_mask(th)
    Y_refined.append(th)

print(X_test.dtype, X_test.shape)



Y_th = []
coord_for_AT = []
for i in range(len(Y_refined)):
    cnts = cv2.findContours(Y_refined[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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










X_masked = []
print('Applying masks...')
for i in tqdm(range(len(Y_test))):
    img = X_test[i][:,:,0].astype(np.uint8)
    mask = Y_th[i]
    img = cv2.bitwise_or(img, img, mask=mask)
    #img_temp = cv2.bitwise_or(img, img)
    X_masked.append(img)
    
    
X_affine = []
print('Affine transforming masks...')
for i in tqdm(range(len(Y_test))):
    img = X_masked[i]
    pts1 = np.float32(coord_for_AT[i])
    pts2 = np.float32([(0,0), (128,128), (0, 128)])
    matrix = cv2.getAffineTransform(pts1, pts2)
    affine = cv2.warpAffine(img, matrix, (len(img), len(img)))
    X_affine.append(affine)

X_thresholded = []
print('Applying threshold...')
for i in tqdm(range(len(Y_test))):
    img = X_masked[i]
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.erode(img, kernel, 3) 
    ret, img = cv2.threshold(img, img.max()*0.8, 255, cv2.THRESH_BINARY)
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 30)
    #img = cv2.bitwise_not(img)
    X_thresholded.append(img)


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
        Save_images(current_img)