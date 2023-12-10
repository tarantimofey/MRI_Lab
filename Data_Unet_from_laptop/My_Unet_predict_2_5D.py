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
from brukerapi.dataset import Dataset

import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import os
import random
from tqdm import tqdm

import plotly.graph_objects as go

from skimage.io import imread, imshow
from skimage.transform import resize

#import segmentation_models_3D as sm

from tensorflow.keras import backend as K 
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    UpSampling3D,
    BatchNormalization,
    PReLU,
    SpatialDropout3D,   
    Dropout,
    ReLU,
    concatenate,
    #Deconvolution3D,    
)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


from tensorflow.keras import layers

#from keras.layers import Deconvolution3D,  

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_DEPTH = 128
IMG_CHANNELS = 1


paths = []
paths.append('C:/Users/Taran/Desktop/rat_5/LUNGS.h64/2/pdata/1/') #0

paths.append('C:/Users/Taran/Desktop/rat_5/LUNGS.ha4/3/pdata/1/') #1

paths.append('C:/Users/Taran/Desktop/rat_5/LUNGS.hh3/2/pdata/1/') #2
paths.append('C:/Users/Taran/Desktop/rat_5/LUNGS.hh3/3/pdata/1/') #3

paths.append('C:/Users/Taran/Desktop/rat_5/LUNGS.ho3/3/pdata/1/') #4
paths.append('C:/Users/Taran/Desktop/rat_5/LUNGS.ho3/4/pdata/1/') #5
path = paths[2]


#model = tf.keras.models.load_model('C:/Users/Taran/Desktop/2_5D_Unet.h5')
#model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/rat5/1Run/rat5_2.h5')
model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/rat5/1Run/rat5_5.h5')


rotation = 1

def Show_XY_image(n):
    global XY_img 
    XY_img  = d[:, :, n]
    XY_img = cv2.normalize(XY_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    #cv2.namedWindow('XY', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow("XY", XY_img)
    
    XY_img_color = XY_img.copy()
    XY_img_color = cv2.cvtColor(XY_img_color, cv2.COLOR_GRAY2RGB)
    
    XY_mask = mask[:, :, n]
    XY_mask = cv2.normalize(XY_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    XY_mask_color = XY_mask.copy()
    XY_mask_color = cv2.cvtColor(XY_mask_color, cv2.COLOR_GRAY2RGB)
    
    for i in range(0, len(XY_mask_color)):
        for j in range(0, len(XY_mask_color)):
            XY_mask_color[i][j][0] = 0
            XY_mask_color[i][j][1] = 0   
    superimposed = cv2.addWeighted(XY_img_color, 1, XY_mask_color, 0.5, 0)
    cv2.namedWindow('XY+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("XY+mask", superimposed)  
    cv2.resizeWindow("XY+mask", 853, 853)
        
def Show_XZ_image(n):
    current_img_XZ = n
    XZ_img = d[:, n, :]
    XZ_img = cv2.normalize(XZ_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    #cv2.namedWindow('XZ', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow("XZ", XZ_img)
    
    XZ_img_color = XZ_img.copy()
    XZ_img_color = cv2.cvtColor(XZ_img_color, cv2.COLOR_GRAY2RGB)
    
    XZ_mask = mask[:, n, :]
    XZ_mask = cv2.normalize(XZ_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    XZ_mask_color = XZ_mask.copy()
    XZ_mask_color = cv2.cvtColor(XZ_mask_color, cv2.COLOR_GRAY2RGB)
    
    for i in range(0, len(XZ_mask_color)):
        for j in range(0, len(XZ_mask_color)):
            XZ_mask_color[i][j][0] = 0
            XZ_mask_color[i][j][1] = 0   
    superimposed = cv2.addWeighted(XZ_img_color, 1, XZ_mask_color, 0.5, 0)
    cross_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
    cross_img = cv2.cvtColor(cross_img, cv2.COLOR_GRAY2RGB)
    cross_img = cv2.rectangle(cross_img,(current_img_XY,0),(current_img_XY,IMG_HEIGHT),(0,255,0),-1)
    cross_img = cv2.rectangle(cross_img,(0, current_img_YZ),(IMG_WIDTH, current_img_YZ),(255,0,0),-1)
    #superimposed = cv2.addWeighted(superimposed, 1, cross_img, 0.2, 0)
    cv2.namedWindow('XZ+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("XZ+mask", superimposed)  
    cv2.resizeWindow("XZ+mask", 853, 853)

def Show_YZ_image(n):
    current_img_YZ = n
    YZ_img = d[n, :, :]
    YZ_img = cv2.normalize(YZ_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    #cv2.namedWindow('YZ', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow("YZ", YZ_img)
    
    YZ_img_color = YZ_img.copy()
    YZ_img_color = cv2.cvtColor(YZ_img_color, cv2.COLOR_GRAY2RGB)
    
    YZ_mask = mask[n, :, :]
    YZ_mask = cv2.normalize(YZ_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    YZ_mask_color = YZ_mask.copy()
    YZ_mask_color = cv2.cvtColor(YZ_mask_color, cv2.COLOR_GRAY2RGB)
    
    for i in range(0, len(YZ_mask_color)):
        for j in range(0, len(YZ_mask_color)):
            YZ_mask_color[i][j][0] = 0
            YZ_mask_color[i][j][1] = 0   
    superimposed = cv2.addWeighted(YZ_img_color, 1, YZ_mask_color, 0.5, 0)
    cross_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
    cross_img = cv2.cvtColor(cross_img, cv2.COLOR_GRAY2RGB)
    cross_img = cv2.rectangle(cross_img,(current_img_XY,0),(current_img_XY,IMG_HEIGHT),(0,255,0),-1)
    cross_img = cv2.rectangle(cross_img,(0, current_img_XZ),(IMG_WIDTH, current_img_XZ),(255,0,0),-1)
    #superimposed = cv2.addWeighted(superimposed, 1, cross_img, 0.2, 0)
    cv2.namedWindow('YZ+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("YZ+mask", superimposed)     
    cv2.resizeWindow("YZ+mask", 853, 853)




dataset = Dataset(path+'2dseq')    # create data set, works for fid, 2dseq, rawdata.x, ser
#X = dataset.data                         # access data array
#dataset.VisuCoreSize                 # get a value of a single parameter
d = np.asarray(dataset.data)
d = dataset.data[:, :, :, 0]
print('dataset loaded')


if rotation == 1:
    d = np.rot90(d, 1, (0,1))
    d = np.flip(d, 0)


#46:110
#14:142
#d = d[14:142, 14:142, 46:110]
d = d[14:142, 14:142, 14:142]


X_test = []

for i in tqdm(range(len(d))):
    x = d[:,:,i]
    x = x[..., np.newaxis]
    X_test.append(x)

X_test = np.asarray(X_test)
#X_test = np.asarray(X_test)
print(X_test.shape)

Y_test = model.predict(X_test, verbose=1)
Y_test = (Y_test > 0.5).astype(np.uint8)

with np.printoptions(threshold=np.inf):
    print(Y_test[60])
#d = d[:, :, :, 0].astype(np.float32)
#mask = Y_test[:, :, :, 0].astype(np.uint8)
#Y_test *= 255
mask = []
for i in range(len(Y_test)):
    mask.append(Y_test[i])
mask = np.asarray(mask)
mask = np.swapaxes(mask,0,2)

mask = np.rot90(mask, 1, (0,1))
mask = np.flip(mask, 0)

"""
with open('C:/Users/Taran/Desktop/mask_predict.txt', 'w') as f:
    with np.printoptions(threshold=np.inf):
        f.write(str(mask))
        #np.save(f, mask)
        print(mask)
print('mask_predict.txt saved!')
"""

print(d.shape, mask.shape)



img_XY = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
img_XZ = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
img_YZ = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)


current_img = 0
current_img_XY = 0
current_img_XZ = 0
current_img_YZ = 0
img_len = len(X_test)
Show_XY_image(current_img_XY)
Show_XZ_image(current_img_XZ)
Show_YZ_image(current_img_YZ)

"""
cv2.setMouseCallback("XY+mask", Mouse_draw_XY)
cv2.setMouseCallback("XZ+mask", Mouse_draw_XZ)
cv2.setMouseCallback("YZ+mask", Mouse_draw_YZ)
"""

view = 0

while(1):
    full_key_code = cv2.waitKeyEx(0)
    if full_key_code == 119: #2555904 W/right
        current_img_XY += 1
        if current_img_XY >= img_len:
            current_img_XY = img_len-1          
        Show_XY_image(current_img_XY)
        print('image XY ' + str(current_img_XY) + ' / ' + str(img_len))
    if full_key_code == 113: #2424832 Q/left
        current_img_XY -= 1        
        if current_img_XY < 0:
            current_img_XY = 0
        Show_XY_image(current_img_XY)
        print('image XY ' + str(current_img_XY) + ' / ' + str(img_len))
    if full_key_code == 115: #S
        current_img_XZ += 1
        if current_img_XZ >= img_len:
            current_img_XZ = img_len-1          
        Show_XZ_image(current_img_XZ)
        print('image XZ ' + str(current_img_XZ) + ' / ' + str(img_len))
    if full_key_code == 97: #A
        current_img_XZ -= 1        
        if current_img_XZ < 0:
            current_img_XZ = 0
        Show_XZ_image(current_img_XZ)
        print('image XZ ' + str(current_img_XY) + ' / ' + str(img_len))
    if full_key_code == 120: #X
        current_img_YZ += 1
        if current_img_YZ >= img_len:
            current_img_YZ = img_len-1          
        Show_YZ_image(current_img_YZ)
        print('image YZ ' + str(current_img_YZ) + ' / ' + str(img_len))
    if full_key_code == 122: #Z
        current_img_YZ -= 1        
        if current_img_YZ < 0:
            current_img_YZ = 0
        Show_YZ_image(current_img_YZ)
        print('image YZ ' + str(current_img_YZ) + ' / ' + str(img_len))
    if full_key_code == 109: #M
        X, Y, Z = np.mgrid[0:128, 0:128, 0:128]
        fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=mask.flatten(),
        isomin=0.1,
        isomax=10,
        opacity=1, # needs to be small to see through all surfaces
        surface_count=20, # needs to be a large number for good volume rendering
        ))
        fig.show()
    if full_key_code == 118: #V
        if view == 0:
            n = current_img_XY
            XY_img  = d[:, :, n]
            XY_img = cv2.normalize(XY_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
            cv2.imshow("XY+mask", XY_img)
            n = current_img_XZ
            XZ_img  = d[:, n, :]
            XZ_img = cv2.normalize(XZ_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
            cv2.imshow("XZ+mask", XZ_img)
            n = current_img_YZ
            YZ_img  = d[n, :, :]
            YZ_img = cv2.normalize(YZ_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
            cv2.imshow("YZ+mask", YZ_img)
            view = 1   
        elif  view == 1:    
            Show_XY_image(current_img_XY)
            Show_XZ_image(current_img_XZ)
            Show_YZ_image(current_img_YZ)
            view = 0
    if full_key_code == 32:
        """
        cv2.imwrite('C:/Users/Taran/Desktop/' + 'Img' + str(n) + '.png', superimposed)
        print('image ' + str(current_img) + ' saved!')
        """
        #cv2.imwrite('C:/Users/Taran/Desktop/' + 'VR_Data.png', VR_Data)
        #print('VR_Data.png saved!')
        """
        with open(path+'mask.npy', 'wb') as f:
            with np.printoptions(threshold=np.inf):
                #f.write(str(mask))
                np.save(f, mask)
        print('mask.npy saved!')
        """
    #plt.show(block=False)
    if full_key_code == 114: #R
        """
        with open(path+'mask.npy', 'rb') as f:
            with np.printoptions(threshold=np.inf):
                #mask = f.read()
                mask = np.load(f)
        print('mask.npy loaded')
        """
    #plt.show(block=False)