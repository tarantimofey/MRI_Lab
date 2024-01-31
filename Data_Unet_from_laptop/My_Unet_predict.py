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

import os
import random
from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.metrics import jaccard_score


import argparse
from pathlib import Path


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

X_path = ''
Y_path = ''
model_path = ''
Y_pat_path = ''
model_pat_path = ''

parser = argparse.ArgumentParser()
parser.add_argument("--X_path", help="enter X path")
parser.add_argument("--Y_path", help="enter Y path")
parser.add_argument("--model_path", help="enter model path")
parser.add_argument("--Y_pat_path", help="enter Y path")
parser.add_argument("--model_pat_path", help="enter model path")
args = parser.parse_args()
if args.X_path:     
    if not os.path.exists(args.X_path):
        print('No valid X path entered!')
        exit()
    X_path = os.path.normpath(args.X_path) + '/'
else:
    print('X_path == none')
    exit()
if args.Y_path:     
    if not os.path.exists(args.Y_path):
        print('No valid X path entered!')
        exit()
    Y_path = os.path.normpath(args.Y_path) + '/'
else:
    print('Y_path == none')
    exit()
if args.model_path:     
    if not os.path.exists(args.model_path):
        print('No valid X path entered!')
        exit()
    model_path = os.path.normpath(args.model_path) + '/'
else:
    print('model_path == none')
    exit()
if args.Y_pat_path:     
    if not os.path.exists(args.Y_pat_path):
        print('No valid X path entered!')
        exit()
    Y_pat_path = os.path.normpath(args.Y_pat_path) + '/'
if args.model_pat_path:     
    if not os.path.exists(args.model_pat_path):
        print('No valid X path entered!')
        exit()
    model_pat_path = os.path.normpath(args.model_pat_path) + '/'

import tensorflow as tf
"""
#path = 'C:/Users/Taran/Desktop/Data_Unet/TEST/Image/'
path = 'C:/Users/Taran/Desktop/Ready_data/OP_SS/Png/Image/'
#path = 'C:/Users/Taran/Desktop/Ready_data/OP_NS/Png/Image/'
#path = 'C:/Users/Taran/Desktop/Ready_data/OP_15/Png/Image/'
#path = 'C:/Users/Taran/Desktop/anonimus/Png/Image/'

# path = 'K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/21rat/Png/EXP_DATA/'
# path_ground = 'K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/21rat/Png/MASK_EXP/'
# path = 'K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/Png/EXP_DATA/'
# path_ground = 'K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/Png/MASK_EXP/'
path = '/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/EXP_DATA/'
path_ground = '/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/MASK_PAT/'
model = tf.keras.models.load_model('/media/taran/SSD2/Data_Unet_from_laptop/Models/Olya_26_07_23/PAT-2Run/Pat.h5')
"""
model = tf.keras.models.load_model(model_path)


def Show_images(n):

    #global X_img
    #global Y_img
    #global Y_img_refined
    #global Y_img_ground
    #global X_masked
    #global X_affine
    #global X_thresholded
    #global Y_img_pat_ground
    #global Y_img_pat
    
    X_img = X_test[n][:, :, 0]
    Y_img = Y_test[n][:, :, 0]    
    if args.model_pat_path:
        Y_img_pat = Y_pat_affine[n]


    Y_img_refined = Y_refined[n][:, :]
    Y_img_ground = Y_ground[n]
    X_img_masked = X_masked[n][:, :]
    X_img_affine = X_affine[n][:, :]
    X_img_thresholded = X_thresholded[n]
    if args.Y_pat_path:
        Y_img_pat_ground = Y_pat_ground[n]
        
    X_img = cv2.normalize(X_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img = cv2.normalize(Y_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img_ground = cv2.normalize(Y_img_ground, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img_ground *= 255
    #Y_img_refined = cv2.normalize(Y_img_refined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    X_img_masked = cv2.normalize(X_img_masked, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    X_img_affine = cv2.normalize(X_img_affine, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    X_img_thresholded = cv2.normalize(X_img_thresholded, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    if args.Y_pat_path:
        Y_img_pat_ground = cv2.normalize(Y_img_pat_ground, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        Y_img_pat_ground *= 255
    if args.model_pat_path:
        Y_img_pat = cv2.normalize(Y_img_pat, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        Y_img_pat *= 255



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
    if args.Y_pat_path:
        cv2.namedWindow('Y_pat_ground', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Y_pat_ground", Y_img_pat_ground)
        #cv2.resizeWindow('Y', 200, 200)  
    if args.model_pat_path:  
        cv2.namedWindow('Y_pat', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Y_pat", Y_img_pat)
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
    
    if args.Y_pat_path:
        superimposed_pat_ground = Superimpose(X_img, Y_img_pat_ground)
        cv2.namedWindow('Img+mask_pat_ground', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Img+mask_pat_ground", superimposed_pat_ground)

    if args.Y_pat_path:
        superimposed_pat = Superimpose(X_img, Y_img_pat)
        cv2.namedWindow('Img+mask_pat', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Img+mask_pat", superimposed_pat)


    print('Dice_coeff = ', Dice_coeff(Y_img, Y_img_ground))
    if args.Y_pat_path:
        print('Dice_coeff_pat = ', Dice_coeff(Y_img_pat, Y_img_pat_ground))
    print('\n')

    if args.Y_pat_path:
        print('IoU_coeff_pat = ', jaccard_score(Y_img_pat_ground, Y_img_pat, average='micro'))
    print('\n')

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
    #dice = np.sum(seg[gt==k]==k)*2.0 / (np.sum(seg[seg==k]==k) + np.sum(gt[gt==k]==k))
    dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))
    return dice

"""
test_ids = []
temp = listdir(X_path)
temp.sort(key=len)
test_ids = temp
print(test_ids)
"""

test_ids = listdir(X_path)
#test_ids.sort(key=len)
test_ids.sort(key=lambda x: (len(x), x))
print(test_ids)

#test_ids = [ f for f in listdir(path) if isfile(join(path,f)) ]
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
print('\nResizing training images and masks...')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = imread(X_path + id_)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img[..., np.newaxis]
    X_test[n] = img
print('Done!\n')

Y_ground = []
print('\nResizing training images and masks...')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    if os.path.exists(Y_path + id_):
        img = imread(Y_path + id_)[:,:]
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
    img = X_test[i][:,:,0]
    mask = Y_th[i]
    img = cv2.bitwise_or(img, img, mask=mask)
    #img_temp = cv2.bitwise_or(img, img)
    X_masked.append(img)
    
    
X_affine = []
print('Affine transforming X_masked...')
for i in tqdm(range(len(X_masked))):
    img = X_masked[i]
    pts1 = np.float32(coord_for_AT[i])
    pts2 = np.float32([(0,0), (128,128), (0, 128)])
    matrix = cv2.getAffineTransform(pts1, pts2)
    affine = cv2.warpAffine(img, matrix, (len(img), len(img)))
    X_affine.append(affine)

X_thresholded = []
print('Applying threshold...')
for i in tqdm(range(len(X_masked))):
    img = X_masked[i]
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.erode(img, kernel, 3) 
    ret, img = cv2.threshold(img, img.max()*0.8, 255, cv2.THRESH_BINARY)
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 30)
    #img = cv2.bitwise_not(img)
    X_thresholded.append(img)



Dice_coefficients = []
for i in range(len(Y_ground)):
    if Y_ground[i].any() != 0:
        Y_img_ground = Y_ground[i]
        Y_img_ground = cv2.normalize(Y_img_ground, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        Y_img_ground *= 255
        Y_img = Y_test[i][:,:,0]
        Y_img = cv2.normalize(Y_img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        Y_img *= 255
        Dice_coefficients.append(Dice_coeff(Y_img, Y_img_ground))
Dice_coefficients = np.asarray(Dice_coefficients)
#print('Dice_coefficients = ', Dice_coefficients)

print('Dice_coefficients: mean =', Dice_coefficients.mean(), ', min =', Dice_coefficients.min(), ', max =', Dice_coefficients.max())











 
Y_pat_ground = []
if args.Y_pat_path:    
    print('\nResizing training images and masks...')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        if os.path.exists(Y_pat_path + id_):
            img = imread(Y_pat_path + id_)[:,:]
            #img = cv2.imread(Y_pat_path + id_, 2)
            #img = img[..., np.newaxis]
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            #print(img.dtype)
            Y_pat_ground.append(img)
        else:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            Y_pat_ground.append(img)
            
    print('Done!\n')


if args.model_pat_path:
    model_pat = tf.keras.models.load_model(model_pat_path)
    X_affine = np.asarray(X_affine)
    X_affine = X_affine[..., np.newaxis]
    Y_pat = model_pat.predict(X_affine, verbose=1)
    Y_pat = np.asarray(Y_pat)
    print(Y_pat.shape)
    Y_pat_affine = []
    for i in range(len(Y_pat)):
        img = Y_pat[i][:,:,0]
        pts1 = np.float32(coord_for_AT[i])
        pts2 = np.float32([(0,0), (128,128), (0, 128)])
        matrix = cv2.getAffineTransform(pts2, pts1)
        affine = cv2.warpAffine(img, matrix, (len(img), len(img)))
        Y_pat_affine.append(affine)

    Y_pat_affine = np.asarray(Y_pat_affine)
    Y_pat_ground = np.asarray(Y_pat_ground)
    print(Y_pat_affine.shape, Y_pat_ground.shape)
    Dice_coefficients_pat = []
    for i in range(len(Y_pat_ground)):
        if Y_pat_ground[i].any() != 0:
            Y_img_pat = Y_pat_affine[i]
            Y_img_pat_ground = Y_pat_ground[i]
            Y_img_pat_ground = cv2.normalize(Y_img_pat_ground, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
            Y_img_pat_ground *= 255
            Y_img_pat = cv2.normalize(Y_img_pat, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
            Y_img_pat *= 255
            Dice_coefficients_pat.append(Dice_coeff(Y_img_pat, Y_img_pat_ground))
    Dice_coefficients_pat = np.asarray(Dice_coefficients_pat)

    IoU_coefficients_pat = []
    for i in range(len(Y_pat_ground)):
        if Y_pat_ground[i].any() != 0:
            Y_img_pat = Y_pat_affine[i]
            Y_img_pat_ground = Y_pat_ground[i]
            Y_img_pat_ground = cv2.normalize(Y_img_pat_ground, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
            Y_img_pat_ground *= 255
            Y_img_pat = cv2.normalize(Y_img_pat, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
            Y_img_pat *= 255
            IoU_coefficients_pat.append(jaccard_score(Y_img_pat_ground, Y_img_pat, average='micro'))
    IoU_coefficients_pat = np.asarray(IoU_coefficients_pat)
    #print('IoU_coefficients_pat = ', IoU_coefficients_pat)
    print('IoU_coefficients_pat: mean =', IoU_coefficients_pat.mean(), ', min =', IoU_coefficients_pat.min(), ', max =', IoU_coefficients_pat.max())






with open("/home/taran/Rat_Unet/lung.txt", 'a') as file:
    writestr = str(Dice_coefficients.mean()) + ', ' + str(Dice_coefficients.min()) + ', ' + str(Dice_coefficients.max()) + '\n'
    file.write(writestr)











"""
import imageio
for i in range(len(Y_test_pat)):
    cv2.imwrite('/home/taran/aa/img' + str(i) + '.png', cv2.normalize(Y_test_pat[i][:,:,0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
"""









current_img = 0
Show_images(current_img)


cv2.moveWindow('X', 1, 0)
cv2.moveWindow('Y', 400, 0)
cv2.moveWindow('Y_refined', 800, 0)
cv2.moveWindow('Y_ground', 1200, 0)

cv2.moveWindow('X_masked', 1, 400)
cv2.moveWindow('X_affine', 400, 400)
cv2.moveWindow('X_thresholded', 800, 400)
if args.Y_pat_path:
    cv2.moveWindow('Y_pat_ground', 1200, 400)
if args.model_pat_path:
    cv2.moveWindow('Y_pat', 1600, 400)

cv2.moveWindow('Img+mask', 1, 800)
cv2.moveWindow('Img+mask_refined', 400, 800)
cv2.moveWindow('Img+mask_ground', 800, 800)
cv2.moveWindow('Img+mask_pat_ground', 1200, 800)
cv2.moveWindow('Img+mask_pat', 1600, 800)











img_len = len(X_test)
while(1):
    full_key_code = cv2.waitKeyEx(0)
    if full_key_code == 65363:
        current_img += 1
        if current_img >= img_len:
            current_img = img_len-1
        print('image ' + str(current_img) + ' / ' + str(img_len))
        Show_images(current_img)
    if full_key_code == 65361:
        current_img -= 1        
        if current_img < 0:
            current_img = 0
        print('image ' + str(current_img) + ' / ' + str(img_len))
        Show_images(current_img)
    if full_key_code == 32:
        Save_images(current_img)
    if full_key_code == 27:
            exit()