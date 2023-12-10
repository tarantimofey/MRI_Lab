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
from sklearn.utils import shuffle
import os
import random
from tqdm import tqdm

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
path = paths[0]



def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), 
                   epsilon=0.00001):
    """
    Compute mean soft dice loss over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator in formula for dice loss.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_loss (float): computed value of dice loss.     
    """

    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true**2, axis=axis) + K.sum(y_pred**2, axis=axis) + epsilon
    dice_loss = 1 - K.mean((dice_numerator)/(dice_denominator))

    return dice_loss



dataset = Dataset(path+'2dseq')    # create data set, works for fid, 2dseq, rawdata.x, ser
#X = dataset.data                         # access data array
#dataset.VisuCoreSize                 # get a value of a single parameter
d = np.asarray(dataset.data)
d = dataset.data[:, :, :, 0]
print('dataset loaded')

with open(path+'mask.npy', 'rb') as f:
    with np.printoptions(threshold=np.inf):
        #mask = f.read()
        mask = np.load(f)
print('mask.npy loaded')
mask = mask//255

#46:110
#14:142
"""
d = d[14:142, 14:142, 14:142]
mask = mask[14:142, 14:142, 14:142]
"""
#51:76 81:108

dl = d[14:142, 14:142, 51:76]
dr = d[14:142, 14:142, 81:108]
d = np.concatenate((dl, dr), axis=2)
maskl = mask[14:142, 14:142, 51:76]
maskr = mask[14:142, 14:142, 81:108]
mask = np.concatenate((maskl, maskr), axis=2)

"""
with np.printoptions(threshold=np.inf):
    print(mask)
"""
#X_test = []
X_train = []
Y_train = []
#X_train = d[..., np.newaxis]
#Y_train = mask[..., np.newaxis]

for i in tqdm(range(25)):
    x = d[:,:,i]
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    x = x[..., np.newaxis]
    X_train.append(x)
    y = mask[:,:,i]
    y = y[..., np.newaxis]
    Y_train.append(y)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
#X_test = np.asarray(X_test)

#X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 1)
X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

print(X_train.shape, Y_train.shape)


def Show_XY_image(n):
    global XY_img 
    XY_img  = X_train[n, :, :, 0]
    XY_img = cv2.normalize(XY_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    #cv2.namedWindow('XY', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow("XY", XY_img)
    
    XY_img_color = XY_img.copy()
    XY_img_color = cv2.cvtColor(XY_img_color, cv2.COLOR_GRAY2RGB)
    
    XY_mask = Y_train[n, :, :, 0]
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
        


img_XY = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

current_img = 0
current_img_XY = 0
img_len = len(X_train)
Show_XY_image(current_img_XY)




def unet_2_5D(vol_size):
    inputs = tf.keras.layers.Input(shape=vol_size)
    s = tf.keras.layers.Lambda(lambda x: x/1.0)(inputs)


    #Contracting path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

    c3 = tf.keras.layers.Conv2D(63, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(63, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)


    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.2)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expanding path
    u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.2)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.2)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    #model.summary()
    return model








#model = unet_2_5D((IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
model = unet_2_5D((128,128,IMG_CHANNELS))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model_file = 'C:/Users/Taran/Desktop/2_5D_Unet.h5'
checkpointer = tf.keras.callbacks.ModelCheckpoint(model_file, verbose=1, save_best_only='True')
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='C:/Users/Taran/Desktop/logs')
    #tf.keras.callbacks.LearningRateScheduler(scheduler)
]

results = model.fit(X_train,Y_train, validation_split=0.2, batch_size=16, epochs=100, callbacks=[callbacks, checkpointer])


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

