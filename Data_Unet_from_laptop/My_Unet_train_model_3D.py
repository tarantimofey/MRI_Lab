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

IMG_WIDTH = 156
IMG_HEIGHT = 156
IMG_DEPTH = 156
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



dataset = Dataset(paths[0]+'2dseq')    # create data set, works for fid, 2dseq, rawdata.x, ser
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
mask = mask%1

#46:110
#14:142
#d = d[14:142, 14:142, 46:110]
#mask = mask[14:142, 14:142, 46:110]
d = d[14:142, 14:142, 14:142]
mask = mask[14:142, 14:142, 14:142]
"""
with np.printoptions(threshold=np.inf):
    print(mask[:,:,34])
"""    

#X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask_cat, test_size = 0.10, random_state = 0)
#X_test = []
Y_train = []
X_train = []


X_train.append(d[..., np.newaxis])
Y_train.append(mask[..., np.newaxis])
#X_test.append(d1[..., np.newaxis])
Y_train = np.asarray(Y_train)
X_train = np.asarray(X_train)
#X_test = np.asarray(X_test)
print(X_train.shape, Y_train.shape)




"""
def create_convolution_block(input_layer, n_filters, batch_normalization=False,
                             kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1),
                             instance_normalization=False):
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(
        input_layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)
        
def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2),
                       strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Conv3DTranspose(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

def unet_model_3d(loss_function, input_shape=(4, 160, 160, 16),
                  pool_size=(2, 2, 2), n_labels=3,
                  initial_learning_rate=0.00001,
                  deconvolution=False, depth=4, n_base_filters=32,
                  include_label_wise_dice_coefficients=False, metrics=[],
                  batch_normalization=False, activation_name="sigmoid"):

    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer,
                                          n_filters=n_base_filters * (
                                                  2 ** layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1,
                                          n_filters=n_base_filters * (
                                                  2 ** layer_depth) * 2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth - 2, -1, -1):
        
        #print(K.int_shape(current_layer)[1])
        up_convolution = get_up_convolution(pool_size=pool_size,
                                            deconvolution=deconvolution,
                                            n_filters=K.int_shape(current_layer)[1])(current_layer)
        
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        
        #print(K.int_shape(levels[layer_depth][1])[1])
        current_layer = create_convolution_block(n_filters= K.int_shape(levels[layer_depth][1])[1],
                                                 input_layer=concat, 
                                                 batch_normalization=batch_normalization)
        
        current_layer = create_convolution_block(n_filters= K.int_shape(levels[layer_depth][1])[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_function,
                  metrics=metrics)
    return model
    
    
model = unet_model_3d(loss_function=soft_dice_loss, metrics=[dice_coefficient])



from tensorflow.keras.utils import plot_model
plot_model(model, to_file='U-Net_Model.png')
"""

"""
def get_model(width=128, height=128, depth=64):

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    
    #model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
"""


"""
def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=5, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=True, activation_name="sigmoid"):
    inputs = Input(batch_shape=input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:

            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[4])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=4)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[4],
                                                 input_layer=concat, batch_normalization=batch_normalization, dropout=True)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[4],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization, dropout=True)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    model.summary()
    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False, dropout=False):
    
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=4)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=4)(layer)

    if dropout:
        layer = SpatialDropout3D(rate=0.5)(layer)

    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:

        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

"""


"""
def conv_3d(input_tensor, n_filters, kernel_size = (3, 3, 3), strides = (1, 1, 1),
            activation = "relu", padding = "valid", batch_norm = True, dropout = True):
  '''
  Convolution Block with Batch Normalization, ReLU, and Dropout
  '''
  conv = Conv3D(n_filters, kernel_size, padding = padding, strides = strides)(input_tensor)
  
  if batch_norm:
    conv = BatchNormalization()(conv)
    
  if activation.lower() == "relu":
    conv = ReLU()(conv)
  
  if dropout:
    conv = Dropout(0.3)(conv)
    
  return conv


def upconv_and_concat(tensor_to_upconv, tensor_to_concat, upconv_n_filters, 
                      kernel_size = (2, 2, 2), strides = (2, 2, 2), padding = "valid"):
  '''
  Upsampling, cropping and concatenation given two tensor.
  '''
  
  upconv = Conv3DTranspose(upconv_n_filters, kernel_size, strides = strides,
                           padding = padding)(tensor_to_upconv)
  
  crop_size = (int(tensor_to_concat.shape[1]) - int(tensor_to_upconv.shape[1])*2) // 2
  cropped = Cropping3D((crop_size, crop_size, crop_size))(tensor_to_concat)
  concat = concatenate([upconv, cropped], axis = 4)
  
  return concat


def unet_3d(input_shape, n_classes, loss, metrics, n_gpus = 1, optimizer = "adam", 
            lr = 0.0001, batch_norm = True, activation = "relu", pool_size = (2, 2, 2)):
  
  # Encoder
  input = Input(input_shape)
  conv1_1 = conv_3d(input, 32, batch_norm = batch_norm, activation = activation)
  conv1_2 = conv_3d(conv1_1, 32, batch_norm = batch_norm, activation = activation)
  pool_1 = MaxPooling3D(pool_size)(conv1_2)
  

  conv2_1 = conv_3d(pool_1, 64, batch_norm = batch_norm, activation = activation)
  conv2_2 = conv_3d(conv2_1, 64, batch_norm = batch_norm, activation = activation)
  pool_2 = MaxPooling3D(pool_size)(conv2_2)
  
  conv3_1 = conv_3d(pool_2, 128, batch_norm = batch_norm, activation = activation)
  conv3_2 = conv_3d(conv3_1, 128, batch_norm = batch_norm, activation = activation)
  pool_3 = MaxPooling3D(pool_size)(conv3_2)
  
  conv4_1 = conv_3d(pool_3, 256, batch_norm = batch_norm, activation = activation)
  conv4_2 = conv_3d(conv4_1, 128, batch_norm = batch_norm, activation = activation)
  
  
  # Decoder
  upconv_5 = upconv_and_concat(conv4_2, conv3_2, 128)
  conv5_1 = conv_3d(upconv_5, 128, batch_norm = batch_norm, activation = activation)
  conv5_2 = conv_3d(conv5_1, 64, batch_norm = batch_norm, activation = activation)
  
  upconv_6 = upconv_and_concat(conv5_2, conv2_2, 64)
  conv6_1 = conv_3d(upconv_6, 64, batch_norm = batch_norm, activation = activation)
  conv6_2 = conv_3d(conv6_1, 32, batch_norm = batch_norm, activation = activation)
  
  upconv_7 = upconv_and_concat(conv6_2, conv1_2, 32)
  conv7_1 = conv_3d(upconv_7, 32, batch_norm = batch_norm, activation = activation)
  conv7_2 = conv_3d(conv7_1, 32, batch_norm = batch_norm, activation = activation)
  
  final_conv = Conv3D(n_classes, kernel_size = (1, 1, 1), padding = "same")(conv7_2)
  
  
  model = Model(input, final_conv)
      
  # Compile  
  if optimizer == "adam":
    adam = Adam(lr = lr)
    
    model.compile(optimizer = adam, loss = loss, metrics = metrics)
  
  else:
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
  
  return model
"""




import tensorflow as tf
from keras.models import Model
from keras.layers import Activation, Input, concatenate, BatchNormalization 
from keras.layers import Conv3D, UpSampling3D, Conv3DTranspose
from keras.layers import add
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras
import numpy as np
#import C:/Users/Taran/Desktop/py

DEPTH = 5
RESIDUAL = True
DEEP_SUPERVISION = True
FILTER_GROW = True
INSTANCE_NORM = True
NUM_CLASS = 1
BASE_FILTER = 16

def myConv(x_in, nf, strides=1, kernel_size = 3):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    x_out = Conv3D(nf, kernel_size=3, padding='same',kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = BatchNormalization()(x_out)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out


def Unet3dBlock(l, n_feat):
    if RESIDUAL:
        l_in = l
    for i in range(2):
        l = myConv(l, n_feat)
    return add([l_in, l]) if RESIDUAL else l


def UnetUpsample(l, num_filters):
    l = UpSampling3D()(l)
    l = myConv(l, num_filters)
    return l


BASE_FILTER = BASE_FILTER

def unet3d(vol_size):
    inputs = Input(shape=vol_size)
    depth = DEPTH
    filters = []
    down_list = []
    deep_supervision = None
    layer = myConv(inputs, BASE_FILTER)
    
    for d in range(depth):
        if FILTER_GROW:
            num_filters = BASE_FILTER * (2**d)
        else:
            num_filters = BASE_FILTER
        filters.append(num_filters)
        layer = Unet3dBlock(layer, n_feat = num_filters)
        down_list.append(layer)
        if d != depth - 1:
            layer = myConv(layer, num_filters*2, strides=2)
        
    for d in range(depth-2, -1, -1):
        layer = UnetUpsample(layer, filters[d])
        layer = concatenate([layer, down_list[d]])
        layer = myConv(layer, filters[d])
        layer = myConv(layer, filters[d], kernel_size = 1)
        
        if DEEP_SUPERVISION:
            if 0< d < 3:
                pred = myConv(layer, NUM_CLASS)
                if deep_supervision is None:
                    deep_supervision = pred
                else:
                    deep_supervision = add([pred, deep_supervision])
                deep_supervision = UpSampling3D()(deep_supervision)
    
    layer = myConv(layer, NUM_CLASS, kernel_size = 1)
    
    if DEEP_SUPERVISION:
        layer = add([layer, deep_supervision])
    layer = myConv(layer, NUM_CLASS, kernel_size = 1)
    x = Activation('softmax', name='softmax')(layer)
        
    model = Model(inputs=[inputs], outputs=[x])
    return model


# Build model.
#model = get_model(width=128, height=128, depth=64)
#model = unet_model_3d(input_shape=(1,128,128,128,1))
#model = unet_3d(input_shape=(1,128,128,128,1), n_classes=1, loss='binary_crossentropy', metrics=['accuracy'], n_gpus = 1,
            #lr = 0.0001, batch_norm = True, pool_size = (2, 2, 2))
model = unet3d((128,128,128,1))
model.summary()


# Compile model.
initial_learning_rate = 0.0001
"""
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
"""
"""
model.compile(
    loss="binary_crossentropy",
    #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #loss='sparse_categorical_crossentropy',
    #optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    optimizer=keras.optimizers.Adam(),
    metrics=["acc"],
)
"""
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""
# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100

model.fit(
    X_train,
    Y_train,
    #validation_data=X_test,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
"""


model_file = 'C:/Users/Taran/Desktop/3D_Unet.h5'
checkpointer = tf.keras.callbacks.ModelCheckpoint(model_file, verbose=1, save_best_only='True')
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='C:/Users/Taran/Desktop/logs')
    #tf.keras.callbacks.LearningRateScheduler(scheduler)
]

results = model.fit(X_train,Y_train, validation_split=0, batch_size=16, epochs=100, callbacks=[callbacks, checkpointer])