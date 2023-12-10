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


PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/1rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/2rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/3rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/10rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/11rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/12rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/15rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/19rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/20rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/21rat/png/EXP_DATA')

PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/22rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/23rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/24rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/26rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/27rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/28rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/29rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/30rat/png/EXP_DATA')
                                                            
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/4rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/5rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/7rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/8rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/9rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/14rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/16rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/17rat/png/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/18rat/png/EXP_DATA')


PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/1rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/2rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/3rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/10rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/11rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/12rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/15rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/19rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/20rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/CONTROL/21rat/png/INSP_DATA')

PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/22rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/23rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/24rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/26rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/27rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/28rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/29rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/30rat/png/INSP_DATA')

PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/4rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/5rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/7rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/8rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/9rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/14rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/16rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/17rat/png/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/18rat/png/INSP_DATA')

#model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/Olya_26_07_23/3Run/MRI_lung.h5')
model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/mct/6Run/MRI_lung.h5')


Common_path = os.path.commonpath(PATH)
Common_path = 'K:\Data_Unet/Rat/Olya_26_07_23/'
Save_path = Common_path + '/Unet_masks_2/'
if not os.path.exists(Save_path):
    os.mkdir(Save_path)
    
#path1 = os.path.splitdrive()
#print(PATH[0][len(Common_path):])

print('\n')

#raise SystemExit





for i in range(len(PATH)):
    print('\n\n' + '#'*os.get_terminal_size()[0])
    test_ids = []
    temp = listdir(PATH[i])
    temp.sort(key=len)
    test_ids = temp
    
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint32)
    print('\nResizing training images and masks in dataset ' + str(i) + '...')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        img = imread(PATH[i] + '/' + id_)[:,:]
        img = img[..., np.newaxis]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
    print('\nPredicting dataset ' + str(i) + '...')
    Y_test = model.predict(X_test, verbose=1)
    Y_test = (Y_test > 0.5).astype(np.uint8)
    Y_test *= 255
    print('Done!')
    
    print('\nSavings masks for dataset ' + str(i) + '...')
    Folder_path = Save_path + PATH[i][len(Common_path):]
    if not os.path.exists(Folder_path):
        os.makedirs(Folder_path)
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        img = Y_test[n][:,:]
        #img = resize(img, (152, 152), mode='constant', preserve_range=True)
        img = img[:,:,0]
        img = cv2.resize(img, (152, 152), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(Folder_path + '/' + id_, img)
    print('Done!')

"""
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.namedWindow('Img+mask_ground', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Img+mask_ground", superimposed_ground)
"""