#C:/Users/Taran/AppData/Local/Programs/Python/Python39/python.exe -i "$(FULL_CURRENT_PATH)"

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
#from scipy.optimize import leastsq
#from sklearn.linear_model import LinearRegression
#from scipy import ndimage
#from scipy import signal
import math 
#import My_Preful_lib.My_Preful_lib as mp
from os import listdir
from os.path import isfile, join, dirname, basename

import cv2
#import imageio

#import tensorflow as tf
import os
import random
import shutup
from tqdm import tqdm


import argparse
from pathlib import Path

from skimage.io import imread, imshow
from skimage.transform import resize
import time

path = ''
model_path = ''

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="enter path")
parser.add_argument("--model", help="enter model")
parser.add_argument("--Q_ROI", help="flag for manually selecting perfusion ROIП")
#parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
args = parser.parse_args()
if args.path:     
    if not os.path.exists(args.path):
        print('No valid path entered!')
        exit()
    path = os.path.normpath(args.path) + '/'
    #print(path)
else:
    print('path = none')
    exit()
if args.model:
    model_n = int(args.model)
    print('model_n == ', model_n)     
    if model_n == 3:
        model_path = '/media/taran/SSD2/Data_Unet_from_laptop/Models/Lung/3Run/MRI_lung_1.h5'
    elif model_n == 4:
        model_path = '/media/taran/SSD2/Data_Unet_from_laptop/Models/Lung/4Run/MRI_lung_1.h5'
    elif model_n == 5:
        model_path = '/media/taran/SSD2/Data_Unet_from_laptop/Models/Lung/5Run/MRI_lung_1.h5'
    else:
        print('Invalid model!')
        exit()
else:
    print('Model unspecified!')
    exit()
print(model_path)

start_time = time.time()
print('Programm start, time:', start_time)

import tensorflow as tf





IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

interpolation = 1
sort_mode = 1

border_data_delition = 0
seq_start = 0
seq_len = 0

matplotlib_stop = 0

"""
#path = 'C:/Users/Taran/Desktop/Ready_data/OP_SS/Png/Image/'
#path = 'C:/Users/Taran/Desktop/Ready_data/OP_SS/Tiff/'
path = 'C:/Users/Taran/Desktop/Ready_data/OP_15/Tiff/'
#path = 'C:/Users/Taran/Desktop/Ready_data/OP_NS/Tiff/'
path = '/media/taran/SSD2/Ready_data/OP_15/Tiff/'
#path = '/media/taran/SSD2/Data_Unet_from_laptop/Data/Human/OP_NS_1800/Tiff/'
path = '/media/taran/SSD2/Data_Unet/Human/PREFUL_1200/olyafirst1200/Tiff/Norm/'
path = '/media/taran/SSD2/Data_Unet/Human/PREFUL_2023/Taran_Timofei/GE_128/Tiff/'
path='/media/taran/SSD2/Data_Unet/Human/PREFUL_2023/Grebneva_Anna/GE_100/Tiff_norm/'
"""



#model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/Lung/3Run/MRI_lung_1.h5')
#model = tf.keras.models.load_model('/media/taran/SSD2/Data_Unet_from_laptop/Models/Lung/4Run/MRI_lung_1.h5')
#model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/Lung/4Run/MRI_lung_1.h5')
#model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/Lung/5Run/MRI_lung_1.h5')

model = tf.keras.models.load_model(model_path)


#######################################################################################################
def Show_images(n):
    true_n = n
    if n >= len(X_affine):
        n = len(X_affine)-1
    X_img = X_affine[n]
    #X_img = X[n]
    #X_img = X_masked[n]
    Y_img = Y_affine[n]
    #Y_img = Y[n]

    X_img = cv2.normalize(X_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img = cv2.normalize(Y_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    cv2.namedWindow('X', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X", X_img)
    #cv2.resizeWindow('X', 200, 200)
    cv2.namedWindow('Y', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Y", Y_img)
    #cv2.resizeWindow('Y', 200, 200)


    Y_img_color = Y_img.copy()
    Y_img_color = cv2.cvtColor(Y_img_color, cv2.COLOR_GRAY2RGB)

    for i in range(0, len(Y_img_color)):
        for j in range(0, len(Y_img_color)):
            Y_img_color[i][j][0] = 0
            Y_img_color[i][j][1] = 0       

    X_img_color = X_img.copy()
    X_img_color = cv2.cvtColor(X_img_color, cv2.COLOR_GRAY2RGB)
    global superimposed
    superimposed  = cv2.addWeighted(X_img_color, 1, Y_img_color, 0.5, 0)

    cv2.namedWindow('Img+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Img+mask", superimposed)
    
    n = true_n
    if n >= len(X_hipass):
        n = len(X_hipass)-1
    X_img_hipass = X_hipass[n]
    #X_img_hipass = X[n]
    X_img_hipass_color = X_img_hipass.copy()
    X_img_hipass_color = cv2.normalize(X_img_hipass_color, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    X_img_hipass_color = cv2.cvtColor(X_img_hipass_color, cv2.COLOR_GRAY2RGB)
    X_sd_th_8bit_color = cv2.cvtColor(X_sd_th_8bit, cv2.COLOR_GRAY2RGB)
    X_sd_th_8bit_color[:,:,0] = 0
    X_sd_th_8bit_color[:,:,1] = 0
    X_img_hipass  = cv2.addWeighted(X_img_hipass_color, 1, X_sd_th_8bit_color, 0.5, 0)
    cv2.namedWindow('X_hipass', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X_hipass", X_img_hipass)
    
    X_img_perf_masked = X_perf_masked[n]
    X_img_perf_masked = cv2.normalize(X_img_perf_masked, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.namedWindow('X_perf_masked', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X_perf_masked", X_img_perf_masked)
    
    
    
    n = true_n
    if n >= len(X_hipass_sorted):
        n = len(X_hipass_sorted)-1
    X_img_perf = X_hipass_sorted[n]
    Y_img_perf = Y_affine_for_perf_sorted[n]

    X_img_perf = cv2.normalize(X_img_perf, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Y_img_perf = cv2.normalize(Y_img_perf, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    cv2.namedWindow('X_perf', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X_perf", X_img_perf)
    #cv2.resizeWindow('X', 200, 200)
    cv2.namedWindow('Y_perf', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Y_perf", Y_img_perf)
    #cv2.resizeWindow('Y', 200, 200)
    

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
    else:
        raise AttributeError
    #th_c = cv2.medianBlur(th_c, 3)
    return th

def Affine_transform(img, points_start, points_end):    
    pts1 = np.float32(points_start)
    pts2 = np.float32([points_start[0], points_start[1], points_end[2]])
    matrix = cv2.getAffineTransform(pts1, pts2)
    affine = cv2.warpAffine(img, matrix, (len(img), len(img)))
    return affine

def Create_VR(img_ex, img_in, img_mid):
    #img_ex.astype(np.float32)
    #img_in.astype(np.float32)
    #img_mid.astype(np.float32)
    #VR = cv2.subtract(cv2.divide(img_mid, img_in), cv2.divide(img_mid, img_ex))
    #VR = np.subtract(np.divide(img_mid, img_in), np.divide(img_mid, img_ex))
    #VR = np.divide(img_mid, img_in)
    
    img_mid_over_in = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    for i in range(IMG_HEIGHT):
        for j in range(IMG_HEIGHT):
            if img_in[i, j] == 0:
                img_mid_over_in[i, j] = 0
            else:
                img_mid_over_in[i,j] = img_mid[i, j] / img_in[i, j]
                
    img_mid_over_ex = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    for i in range(IMG_HEIGHT):
        for j in range(IMG_HEIGHT):
            if img_ex[i, j] == 0:
                img_mid_over_ex[i, j] = 0
            else:
                img_mid_over_ex[i,j] = img_mid[i, j] / img_ex[i, j]
                
    VR = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    for i in range(IMG_HEIGHT):
        for j in range(IMG_HEIGHT):
            VR[i, j] = img_mid_over_in[i, j] - img_mid_over_ex[i, j] 
    
    """
    with np.printoptions(threshold=np.inf):
        print(VR)
    """
    #print(VR_img)
    
    
    
    
    count_positive = 0
    count_negative = 0
    VR_mean = 0
    for i in range(0, len(VR)):
        for j in range(0, len(VR)):
            if VR[i][j]>0:
                count_positive+=1
                VR_mean+=VR[i][j]
            if VR[i][j]<0:
                count_negative+=1
                #VR[i][j] = 0
    #count_coefficient = (count_positive-count_negative)/(2*(count_positive+count_negative)) + 0.5
    #VR_mean = np.mean(VR)
    VR_mean=VR_mean/count_positive
    #STANDART DEVIATION
    sd = 0
    for i in range(0, len(VR)):
        for j in range(0, len(VR)):
            if VR[i][j]>0:
                sd += (VR[i][j] - VR_mean)**2
                #VR[i][j] = 0
    sd=math.sqrt(sd/count_positive)
    #print('\nVR:', count_positive, '/', count_negative, '  ', count_coefficient)
    max = VR.max()
    min = VR.min()
    print('\nVR:', count_positive, '/', count_negative)
    print('Mean:', VR_mean, 'sd:', sd)
    return VR, count_positive, count_negative, VR_mean, sd, min, max

def cos_func(x, D, E, F, C):
    y = D*np.cos(E*x + F) + C
    return y

def cos_func_x2(x, D0, E0, F0, D1, E1, F1, C):
    y = D0*np.cos(E0*x + F0) + D1*np.cos(E1*x + F1) + C
    return y

def Delete_by_index(indeces, *arrays):
    arrays = list(arrays)
    for arr in arrays:
        arr = np.delete(arr, indeces, 0)
    arrays = tuple(arrays)
    return arrays

ids = listdir(path)
X = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
print('\n\nResizing images...')
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    img = imread(path + id_)[:,:]
    #img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img = img[..., np.newaxis]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X[n] = img
"""
import pydicom as dicom
dicom_path = 'C:/Users/Taran/Desktop/PREFUL_1200/olyafirst1200.dcm'
#dicom_path = 'C:/Users/Taran/Desktop/PREFUL_1200/olyasecond1200.dcm'
#dicom_path = 'C:/Users/Taran/Desktop/PREFUL_1200/olyathird1200.dcm'
ds = dicom.dcmread(dicom_path)
X = ds.pixel_array
"""



#print('Done!\n')
init_len = len(X)
border_data = len(X)//200
if border_data_delition == 1:
    X = X[border_data:len(X)-border_data]
#X = X[seq_start:]
if seq_len != 0:
    X = X[0:seq_len]
print('\nBorder slices deleted: ', init_len-len(X))

print('\nPredicting masks...')
Y_pred = model.predict(X, verbose=1)
Y_pred = (Y_pred > 0.5).astype(np.uint8)

#print(type(X[0,0,0]))
#X = X[:, :, :, 0].astype(np.uint8)
#X = X[:, :, :, 0].astype(np.float32)
X = X.astype(np.float32)
#X = X[:, :, :, 0]
Y = Y_pred[:, :, :, 0].astype(np.uint8)

X_indeces = np.arange(len(X))



bad_slices = []
print('\nRefining masks...')
for i in tqdm(range(len(Y_pred))):
    try:
        Y[i] = Refine_mask(Y[i])
    except AttributeError:
        bad_slices.append(i)
        cv2.namedWindow('Error_img', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Error_img", cv2.normalize(X[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        cv2.waitKeyEx(0)

print('bad_slices', bad_slices)
X, Y, X_indeces = Delete_by_index(bad_slices, X, Y, X_indeces)




print('\nDetecting rectangles...')
L_rect = []
R_rect = []
for i in tqdm(range(len(Y))):
    cnt, hierarchy = cv2.findContours(Y[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lung_x_0, lung_y_0, lung_w_0, lung_h_0 = cv2.boundingRect(cnt[0])
    lung_x_1, lung_y_1, lung_w_1, lung_h_1 = cv2.boundingRect(cnt[1])
    if(lung_x_0 > lung_x_1):
        R_rect.append([lung_x_0, lung_y_0, lung_w_0, lung_h_0])
        L_rect.append([lung_x_1, lung_y_1, lung_w_1, lung_h_1])
    else:
        R_rect.append([lung_x_1, lung_y_1, lung_w_1, lung_h_1])
        L_rect.append([lung_x_0, lung_y_0, lung_w_0, lung_h_0])
    #Y[i] = cv2.rectangle(Y[i], (L_rect[i][0], L_rect[i][1]), (L_rect[i][0] + L_rect[i][2], L_rect[i][1] + L_rect[i][3]), 100, -1)
    #Y[i] = cv2.rectangle(Y[i], (R_rect[i][0], R_rect[i][1]), (R_rect[i][0] + R_rect[i][2], R_rect[i][1] + R_rect[i][3]), 200, -1)
    Rect_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    Rect_img = cv2.rectangle(Rect_img, (L_rect[i][0], L_rect[i][1]), (L_rect[i][0] + L_rect[i][2], L_rect[i][1] + L_rect[i][3]), 100, -1)
    Rect_img = cv2.rectangle(Rect_img, (R_rect[i][0], R_rect[i][1]), (R_rect[i][0] + R_rect[i][2], R_rect[i][1] + R_rect[i][3]), 200, -1)
    #Y[i] = cv2.addWeighted(Y[i], 1, Rect_img, 0.5, 0)
    
    
print('\nDeleting bad slices (lung height)...')
bad_slices = []
L_rect = np.asarray(L_rect)
R_rect = np.asarray(R_rect)
lung_height = L_rect[:, 3]

print('X_indeces = ', X_indeces)


fig = plt.figure(figsize=(14, 2), num='Lung height')
#ax = plt.subplot()
ax = plt.subplot(4, 1, (1, 3), label='lung_height')

for i in tqdm(range(len(Y))):
    if L_rect[i][1]+L_rect[i][3] < IMG_HEIGHT/2:
        bad_slices.append(i)
#print('bad_slices', bad_slices)
X_indeces, X, Y, L_rect, R_rect, lung_height = Delete_by_index(bad_slices, X_indeces, X, Y, L_rect, R_rect, lung_height)
"""
X_indeces = np.delete(X_indeces, bad_slices, 0)
X = np.delete(X, bad_slices, 0)
Y = np.delete(Y, bad_slices, 0)
L_rect = np.delete(L_rect, bad_slices, 0)
R_rect = np.delete(R_rect, bad_slices, 0)
lung_height = np.delete(lung_height, bad_slices, 0)
"""
print('bad slices found: ', len(bad_slices))

#Copying data for perfusion

X_indeces_for_perf = X_indeces.copy()
X_for_perf = X.copy()
Y_for_perf = Y.copy()
L_rect_for_perf = L_rect.copy()
R_rect_for_perf = R_rect.copy()
lung_height_for_perf = lung_height.copy()



print('\nFitting images...')
#print(L_rect)
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.set_axisbelow(True)
ax.xaxis.grid(color='0.95', which='both', zorder=0)
plt.ylim(lung_height.min()-2,lung_height.max()+2)
plt.plot(X_indeces, lung_height, 'or', linewidth=0.5, markersize=1, zorder=0)
n_of_deleted_data = 0
amplitude = lung_height.max() - lung_height.min()
mean_lung_height = lung_height.mean()
min_lung_height = lung_height.min()
tolerance_max = amplitude*0.1
tolerance_min_a = amplitude*0.3
tolerance_min_b = amplitude*0.05
tolerance_a = amplitude*0.4
level_1 = amplitude*0.4 + min_lung_height
level_2 = amplitude*0.7 + min_lung_height
bad_slices_flag = 1
plt.plot([X_indeces[0], X_indeces[-1]], [mean_lung_height, mean_lung_height], '-r', linewidth=0.5, markersize=1, zorder=0)
#plt.plot([X_indeces[0], X_indeces[-1]], [level_1, level_1], '-b', linewidth=0.5, markersize=1, zorder=0)
#plt.plot([X_indeces[0], X_indeces[-1]], [level_2, level_2], '-g', linewidth=0.5, markersize=1, zorder=0)
while bad_slices_flag == 1:
    bad_slices_flag = 0
    bad_slices = []
    for i in range(1, len(lung_height)-1):
        #max
        if lung_height[i] - lung_height[i-1] > tolerance_max and lung_height[i] - lung_height[i+1] > tolerance_max and lung_height[i] < level_2 and 1:
            bad_slices_flag = 1
            bad_slices.append(i)
            n_of_deleted_data += 1
        if lung_height[i-1] - lung_height[i] > tolerance_min_a and lung_height[i+1] - lung_height[i] > tolerance_min_a and lung_height[i] < mean_lung_height and 0:
            #bad_slices_flag = 1
            bad_slices.append(i)
            n_of_deleted_data += 1
        #min
        if lung_height[i-1] - lung_height[i] > tolerance_min_b and lung_height[i+1] - lung_height[i] > tolerance_min_b and lung_height[i] > level_1 and 1:
            #bad_slices_flag = 1
            bad_slices.append(i)
            n_of_deleted_data += 1
        #very large shit
        if abs(lung_height[i-1] - lung_height[i]) > tolerance_a and abs(lung_height[i+1] - lung_height[i]) > tolerance_a and 1:
            #bad_slices_flag = 1
            bad_slices.append(i)
            n_of_deleted_data += 1
    X_indeces = np.delete(X_indeces, bad_slices, 0)
    X = np.delete(X, bad_slices, 0)
    Y = np.delete(Y, bad_slices, 0)
    L_rect = np.delete(L_rect, bad_slices, 0)
    R_rect = np.delete(R_rect, bad_slices, 0)
    lung_height = np.delete(lung_height, bad_slices, 0)

print('Slices deleted: ', n_of_deleted_data)


plt.plot(X_indeces, lung_height, 'o-b', linewidth=0.5, markersize=1, zorder=1)
plt.xlabel('Slice number')
plt.ylabel('Lung height')
plt.xlabel('Номер скана')
plt.ylabel('Длина лёгкого')
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.13)
#fig.legend()
#plt.ion()  
#plt.show(block=False)
#plt.pause(0.001)


#FITTING
#guess = [0,0,0,0]
guess = [amplitude, 1, 0, mean_lung_height]
x_from = 0
x_to = x_from + 3
#x_to = x_from + 20
chi_sqr_flag = 0
chi_sqr_critical = 0.02
X_phase = []
segment_divider = []
plotting_phase = []
X_C = []

while chi_sqr_flag == 0:
    x_range = X_indeces[x_from:x_to+1]
    y_range = lung_height[x_from:x_to+1]
    parameters, covariance = curve_fit(cos_func, x_range, y_range, p0=guess, maxfev=5000000)
    fit_cosine = cos_func(x_range, parameters[0], parameters[1], parameters[2], parameters[3])
    chi_sqr = 0
    for i in range(0, len(x_range)):
        chi_sqr += ((lung_height[i+x_from] - fit_cosine[i])**2)/(fit_cosine[i]*len(x_range))
    #print(chi_sqr)
    if chi_sqr > chi_sqr_critical:
        #chi_sqr_flag = 1
        x_to -= 1
        x_range = X_indeces[x_from:x_to+1]
        y_range = lung_height[x_from:x_to+1]
        parameters, covariance = curve_fit(cos_func, x_range, y_range, p0=guess, maxfev=50000, bounds=([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
        parameters[2] = parameters[2]%(2*np.pi)
        fit_cosine = cos_func(x_range, parameters[0], parameters[1], parameters[2], parameters[3])
        ######################CHI_SQR
        chi_sqr = 0        
        for i in range(0, len(x_range)):
            chi_sqr += ((lung_height[i+x_from] - fit_cosine[i])**2)/(fit_cosine[i]*len(x_range))
        print("Segment result", X_indeces[x_from], X_indeces[x_to], chi_sqr, parameters[0], parameters[1], parameters[2])
        ########################PHASE
        for i in range(0, len(x_range)-1):
            #      y = D*np.cos(E*x + F) + C
            #phase = math.acos((fit_cosine[i]-parameters[3])/parameters[0])
            #phase = (x_range[i]-parameters[2])*parameters[1] - ((x_range[i]-parameters[2])*parameters[1])//(2*math.pi)*2*math.pi  
            #phase = x_range[i]*parameters[1] + parameters[2] - ((x_range[i]*parameters[1] + parameters[2])//(2*math.pi))*2*math.pi  #Gold
            if parameters[0] < 0:
                ps = np.pi
            else:
                ps = 0
            phase = (x_range[i]*parameters[1] + parameters[2] + ps)%(2*np.pi)
            #phase = x_range[i]*parameters[1] + parameters[2]
            X_phase.append(phase)
            X_C.append(parameters[3])
        #############################    
        segment_divider.append(X_indeces[x_from])
        x_for_plotting_fit = np.arange(x_range[0], x_range[-1], 0.1)
        fit_cosine = cos_func(x_for_plotting_fit, parameters[0], parameters[1], parameters[2], parameters[3])
        plt.plot(x_for_plotting_fit, fit_cosine, '-', linewidth=1, color='tab:orange') # 00FFEA 22FF00
        plt.plot([x_range[-1], x_range[-1]], [lung_height.min(), lung_height.max()], '-', linewidth=0.5, alpha=0.7, color='#0D2249')
        for i in range(0, len(x_for_plotting_fit)):
            #      y = D*np.cos(E*x + F) + C
            #plotting_phase.append(x_for_plotting_fit[i]*parameters[1] + parameters[2] - ((x_for_plotting_fit[i]*parameters[1] + parameters[2])//(2*math.pi))*2*math.pi)
            if parameters[0] < 0:
                ps = np.pi
            else:
                ps = 0
            plotting_phase.append((x_for_plotting_fit[i]*parameters[1] + parameters[2] + ps)%(2*np.pi))
        x_from = x_to
        x_to = x_from + 4
        #plt.plot(x_range, fit_cosine, '-g', linewidth=1, label='fit')
        
    else:
        x_to += 1
        if x_to >= len(X_indeces):
            """
            if x_to - x_from < 4:
                segment_divider.append(X_indeces[x_from])
                break
            """
            x_to == len(X_indeces)-1
            chi_sqr_flag = 1
            x_range = X_indeces[x_from:x_to+1]
            y_range = lung_height[x_from:x_to+1]
            parameters, covariance = curve_fit(cos_func, x_range, y_range, p0=guess, maxfev=50000)
            ######################CHI_SQR
            chi_sqr = 0        
            for i in range(0, len(x_range)):
                chi_sqr += ((lung_height[i+x_from] - fit_cosine[i])**2)/(fit_cosine[i]*len(x_range))
            #print("Segment result", x_from, x_to, chi_sqr)
            ########################PHASE
            for i in range(0, len(x_range)):
                #phase = x_range[i]*parameters[1] + parameters[2] - ((x_range[i]*parameters[1] + parameters[2])//(2*math.pi))*2*math.pi  
                if parameters[0] < 0:
                    ps = np.pi
                else:
                    ps = 0
                phase = (x_range[i]*parameters[1] + parameters[2] + ps)%(2*np.pi)
                X_phase.append(phase)
                X_C.append(parameters[3])
            segment_divider.append(X_indeces[x_from])
            x_for_plotting_fit = np.arange(x_range[0], x_range[-1], 0.1)
            fit_cosine = cos_func(x_for_plotting_fit, parameters[0], parameters[1], parameters[2], parameters[3])
            plt.plot(x_for_plotting_fit, fit_cosine, '-', linewidth=1, color='tab:orange') # 00FFEA 22FF00
            for i in range(0, len(x_for_plotting_fit)):
                #      y = D*np.cos(E*x + F) + C
                #plotting_phase.append(x_for_plotting_fit[i]*parameters[1] + parameters[2] - ((x_for_plotting_fit[i]*parameters[1] + parameters[2])//(2*math.pi))*2*math.pi)
                if parameters[0] < 0:
                    ps = np.pi
                else:
                    ps = 0
                plotting_phase.append((x_for_plotting_fit[i]*parameters[1] + parameters[2] + ps)%(2*np.pi))
            segment_divider.append(X_indeces[-1])


print('segment_divider', segment_divider)

bad_segments = []
bad_slices = []
for i in range(0, len(segment_divider)-1):
    X_C_index_arr = np.where(X_indeces == segment_divider[i])
    X_C_index = X_C_index_arr[0][0]
    #print(X_C_index)
    #if segment_divider[i+1] - segment_divider[i] < 6 or X_C[X_C_index]-lung_height.mean() > 0.1*(lung_height.max()-lung_height.min()): #change .mean() to parameters[3]
    if segment_divider[i+1] - segment_divider[i] < 7:
        print(i)
        bad_segments.append(i)
        for j in range(len(Y)):
            if X_indeces[j] < segment_divider[i+1] and X_indeces[j] >= segment_divider[i]:
                bad_slices.append(j)
for i in range(len(bad_segments)):
    ax.axvspan(segment_divider[bad_segments[i]], segment_divider[bad_segments[i]+1], facecolor='0.42', alpha=0.5)
print('bad_slices', bad_slices)
print('bad_indeces', X_indeces[bad_slices])

X_indeces = np.delete(X_indeces, bad_slices, 0)
X = np.delete(X, bad_slices, 0)
Y = np.delete(Y, bad_slices, 0)
L_rect = np.delete(L_rect, bad_slices, 0)
R_rect = np.delete(R_rect, bad_slices, 0)
lung_height = np.delete(lung_height, bad_slices, 0)
X_phase = np.delete(X_phase, bad_slices, 0)
X_C = np.delete(X_C, bad_slices, 0)
print('bad segments found: ', len(bad_slices))

#plt.plot(X_indeces, X_C, 'o-', linewidth=0.5, markersize=1, color='gold', zorder=1) #Plot C


ax_phase = plt.subplot(4, 1, 4, label='Exhale', sharex=ax)
ax_phase.set_ylabel('Phase')
ax_phase.set_ylabel('Фаза')
ax_phase.set_xlabel('Номер скана')
ax_phase.yaxis.set_minor_locator(ticker.MultipleLocator(np.pi/4))
ax_phase.yaxis.set_major_locator(ticker.MultipleLocator(np.pi))
ax_phase.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax_phase.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax_phase.set_axisbelow(True)
ax_phase.yaxis.grid(color='0.95', which='major', zorder=0)
ax_phase.xaxis.grid(color='0.95', which='both', zorder=0)
ax_phase.set_yticks([0, np.pi, 2*np.pi])
ax_phase.set_yticklabels(['0', 'π', '2π'])
for i in range(len(bad_segments)):
    ax_phase.axvspan(segment_divider[bad_segments[i]], segment_divider[bad_segments[i]+1], facecolor='0.42', alpha=0.5)

plt.plot(X_indeces, X_phase, 'o', linewidth=0.5, label='data', markersize=2, color='tab:blue', alpha=0.5)
x_for_plotting_phase = np.arange(0, len(plotting_phase)/10, 0.1)
plt.plot(x_for_plotting_phase, plotting_phase, '-', linewidth=0.5, label='data', markersize=2, color='tab:green', alpha=0.5)
for i in range(len(segment_divider)):
    plt.plot([segment_divider[i], segment_divider[i]], [0, 2*math.pi], '-', linewidth=0.5, alpha=0.7, color='#0D2249')
#ax_phase.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))


"""
plt.subplots_adjust(hspace=0)
if matplotlib_stop == 1:
    plt.show()
else:
    plt.show(block=False)
"""





#sort_mode = 1
sort_index_lung_height = np.argsort(lung_height)
sort_index_phase = np.argsort(X_phase)
if sort_mode == 0:
    sort_index = sort_index_lung_height
else:
    sort_index = sort_index_phase
lung_height_sorted = []
X_sorted = []
X_indeces_sorted = []
X_phase_sorted = []
Y_sorted = []
R_rect_sorted = []
L_rect_sorted = []
for i in tqdm(range(len(sort_index))):
    lung_height_sorted.append(lung_height[sort_index[i]])
    X_sorted.append(X[sort_index[i]])
    Y_sorted.append(Y[sort_index[i]])
    R_rect_sorted.append(R_rect[sort_index[i]])
    L_rect_sorted.append(L_rect[sort_index[i]])
    X_indeces_sorted.append(X_indeces[sort_index[i]])
    X_phase_sorted.append(X_phase[sort_index[i]])
#print(lung_height_sorted)
min_lung_height_index = sort_index_lung_height[0]

print('len(X_indeces_sorted)', len(X_indeces_sorted))
print('len(X_phase_sorted)', len(X_phase_sorted))
print('len(lung_height_sorted)', len(lung_height_sorted))


new_indeces = np.arange(len(X))
lung_height_sorted = np.asarray(lung_height_sorted)
sorted_guess = [lung_height_sorted.max()-lung_height_sorted.min(), 1/len(lung_height_sorted), 0, lung_height_sorted.mean()]
sorted_fit_parameters, sorted_fit_covariance = curve_fit(cos_func, new_indeces, lung_height_sorted, p0=sorted_guess, maxfev=50000, bounds=([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
sorted_fit = cos_func(new_indeces, sorted_fit_parameters[0], sorted_fit_parameters[1], sorted_fit_parameters[2], sorted_fit_parameters[3])

print('Finding samples with bad phase...')
sorted_bad_indeces = []
for i in tqdm(range(len(lung_height_sorted))):
    #print(i)
    if abs(lung_height_sorted[i] - sorted_fit[i]) > (lung_height_sorted.max()-lung_height_sorted.min())*0.3:
    #if lung_height_sorted[i] - sorted_fit[i] > 7:
        sorted_bad_indeces.append(i)
print('sorted_bad_indeces found: ', len(sorted_bad_indeces))
fig2 = plt.figure(figsize=(14, 2), num='Lung height sorted')
#ax = plt.subplot()
ax2 = plt.subplot(4, 1, (1, 3), label='lung_height')
plt.plot(new_indeces, lung_height_sorted, 'o-', linewidth=0.5, label='data', markersize=2, color='tab:blue', alpha=0.5)
plt.plot(new_indeces, sorted_fit, '-', linewidth=2, label='data', markersize=2, color='tab:orange', alpha=1)
for i in range(len(sorted_bad_indeces)):    
    plt.plot(new_indeces[sorted_bad_indeces[i]], lung_height_sorted[sorted_bad_indeces[i]], 'o', linewidth=2, label='data', markersize=2, color='tab:red', alpha=1)
ax_phase2 = plt.subplot(4, 1, 4, label='Exhale', sharex=ax2)
ax_phase2.set_ylabel('Phase')
plt.plot(new_indeces, X_phase_sorted, 'o-', linewidth=0.5, label='data', markersize=2, color='tab:blue', alpha=0.5)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.13)
"""
if matplotlib_stop == 1:
    plt.show()
else:
    plt.show(block=False)
"""
new_indeces = np.delete(new_indeces, sorted_bad_indeces, 0)
X_sorted = np.delete(X_sorted, sorted_bad_indeces, 0)
Y_sorted = np.delete(Y_sorted, sorted_bad_indeces, 0)
L_rect_sorted = np.delete(L_rect_sorted, sorted_bad_indeces, 0)
R_rect_sorted = np.delete(R_rect_sorted, sorted_bad_indeces, 0)
lung_height_sorted = np.delete(lung_height_sorted, sorted_bad_indeces, 0)
X_phase_sorted = np.delete(X_phase_sorted, sorted_bad_indeces, 0)



print('\nCompressing images...')###Are indeces right? L_rect_sorted[0] or L_rect[min_lung_height_index] maybe?
points_ex = [L_rect_sorted[min_lung_height_index][0], L_rect_sorted[min_lung_height_index][1]], [L_rect_sorted[min_lung_height_index][0]+L_rect_sorted[min_lung_height_index][2], L_rect_sorted[min_lung_height_index][1]], [L_rect_sorted[min_lung_height_index][0], L_rect_sorted[min_lung_height_index][1]+L_rect_sorted[min_lung_height_index][3]]

#print(points_ex)
X_affine = []
Y_affine = []
for i in tqdm(range(len(X_sorted))):
    points = [L_rect_sorted[0][0], L_rect_sorted[i][1]], [L_rect_sorted[0][0]+L_rect_sorted[0][2], L_rect_sorted[i][1]], [L_rect_sorted[0][0], L_rect_sorted[i][1]+L_rect_sorted[i][3]]
    affine_X_img = Affine_transform(X_sorted[i], points, points_ex)
    affine_Y_img = Affine_transform(Y_sorted[i], points, points_ex)
    X_affine.append(affine_X_img)
    Y_affine.append(affine_Y_img)


print('\nApplying masks...')
end_mask = Y_affine[0].copy()
#end_mask = cv2.rectangle(end_mask, (64,120), (128,128), 0, -1)
#end_mask = cv2.circle(end_mask, (93,122), 12, 0, -1)
#end_mask = cv2.circle(end_mask, (80,100), 20, 0, -1) #1:1199
#kernel = np.ones((3, 3), np.uint8)
#end_mask = cv2.erode(end_mask,kernel,iterations = 1) 
X_masked = []
for i in tqdm(range(len(X_affine))):
    img = cv2.bitwise_or(X_affine[i], X_affine[i], mask=end_mask)
    X_masked.append(img)


print('\n\nComputing VR...')
"""
n = 2
img_ex = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
img_in = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
img_mid = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
for i in range(0, n):
    img_ex = np.add(X_masked[i], img_ex)
    img_in = np.add(X_masked[len(X_masked)-1-i], img_in)
    img_mid = np.add(X_masked[len(X_masked)//2 - n + i], img_mid)
"""

mid_point = len(X_masked)//2
quarter_point = len(X_masked)//4
"""
if sort_mode == 0:
    img_ex = X_masked[0] + X_masked[1] + X_masked[2]
    img_in = X_masked[-1] + X_masked[-2] + X_masked[-3]
    img_mid = X_masked[mid_point] + X_masked[mid_point-1] + X_masked[mid_point+1]
if sort_mode == 1:
    img_ex = X_masked[mid_point] + X_masked[mid_point-1] + X_masked[mid_point+1]
    img_in = X_masked[-1] + X_masked[0] + X_masked[1]
    img_mid = X_masked[quarter_point] + X_masked[quarter_point-1] + X_masked[quarter_point+1]
"""
#n = 3


####zoom()


print('len(X_masked), ', len(X_masked))
n = len(X_masked)//50
#n = 100
if n % 2 == 0:
    n += 1
print('n = ', n)
img_ex = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
img_in = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
img_mid = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
if sort_mode == 1 and interpolation != 2:
    for i in range(-(n//2), (n//2)+1):
        #print(i)
        if interpolation == 1:
            c = (n/2-abs(i))/(n/2)
        else:
            c = 1
        img_ex += X_masked[mid_point + i] * c
        img_in += X_masked[-1 + i] * c
        img_mid += X_masked[quarter_point + i] * c


if interpolation == 2:
    n_interp = 24
    X_masked_interp = ndimage.zoom(X_masked, (n_interp/len(X_masked), 1, 1), mode='grid-wrap')
    lung_height_interp = ndimage.zoom(lung_height_sorted, n_interp/len(X_masked))
    X_phase_interp = ndimage.zoom(X_phase_sorted, n_interp/len(X_masked))
    print('X_masked_interp.shape ', X_masked_interp.shape)
    img_ex = X_masked_interp[n_interp//2]
    img_in = X_masked_interp[0]
    img_mid = X_masked_interp[n_interp//4]
    
    interp_indexes = np.arange(n_interp)
    fig3 = plt.figure(figsize=(14, 2))
    ax3 = plt.subplot(4, 1, (1, 3), label='lung_height')
    plt.plot(interp_indexes, lung_height_interp, 'o-', linewidth=0.5, label='data', markersize=2, color='tab:blue', alpha=0.5)

    ax_phase3 = plt.subplot(4, 1, 4, label='Exhale', sharex=ax3)
    ax_phase3.set_ylabel('Phase')
    plt.plot(interp_indexes, X_phase_interp, 'o-', linewidth=0.5, label='data', markersize=2, color='tab:blue', alpha=0.5)
print(img_ex.max(), img_in.max(), img_mid.max())
print(img_ex.min(), img_in.min(), img_mid.min())
VR_images = np.concatenate((img_ex, img_in, img_mid), 1)
VR_images = cv2.normalize(VR_images, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
"""
cv2.namedWindow('VR_images', cv2.WINDOW_KEEPRATIO)
cv2.imshow("VR_images", VR_images)
cv2.resizeWindow("VR_images", 825, 275)
"""





VR_img, pos, neg, mean, sd, min, max = Create_VR(img_ex, img_in, img_mid)
#VR_img = Create_VR(X_masked[0], X_masked[-1], X_masked[len(X_masked)//2])
VR_img = cv2.bitwise_or(VR_img, VR_img, mask=end_mask)
VR_img = cv2.normalize(VR_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
VR_img = cv2.bitwise_or(VR_img, VR_img, mask=end_mask)
#VR_img = cv2.normalize(VR_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.namedWindow('VR', cv2.WINDOW_KEEPRATIO)
cv2.imshow("VR", VR_img)

"""
with np.printoptions(threshold=np.inf):
    print(VR_img)
"""


#imageio.mimsave('C:/Users/Taran/Desktop/video.gif', X, fps=7) 

#print(VR_img.max())
data_x = 62
data_x_step = 15

data_y = 25
data_y_step = 15

Data_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
Data_img = cv2.rectangle(Data_img,(8,19),(18,108),100,-1)
for i in range(0, 88):
    color = (88-i)*255//88
    Data_img = cv2.rectangle(Data_img,(9,i+20),(17,i+20),color,-1)

Data_img = cv2.putText(Data_img, str(round(max, 3)), (22,data_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
Data_img = cv2.putText(Data_img, str(round(min, 3)), (22,108), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
Data_img = cv2.putText(Data_img, 'pos: '+str(pos), (data_x,data_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
Data_img = cv2.putText(Data_img, 'neg: '+str(neg), (data_x,data_y+data_y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
Data_img = cv2.putText(Data_img, 'mean: '+str(round(mean, 3)), (data_x,data_y+2*data_y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
Data_img = cv2.putText(Data_img, 'sd: '+str(round(sd, 3)), (data_x,data_y+3*data_y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)

Data_img = cv2.rectangle(Data_img,(data_x,data_y+3*data_y_step+6),(data_x+60,data_y+3*data_y_step+6),255,-1)
if sort_mode == 0:
    Data_img = cv2.putText(Data_img, 'lung sort', (data_x,data_y+4*data_y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
if sort_mode == 1:
    Data_img = cv2.putText(Data_img, 'phase sort', (data_x,data_y+4*data_y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
if interpolation == 0:
    Data_img = cv2.putText(Data_img, 'interp off, '+str(n), (data_x,data_y+5*data_y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
if interpolation == 1:
    Data_img = cv2.putText(Data_img, 'interp on, '+str(n), (data_x,data_y+5*data_y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
if interpolation == 2:
    Data_img = cv2.putText(Data_img, 'interp full', (data_x,data_y+5*data_y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
cv2.namedWindow('Data', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Data", Data_img)



VR_Data = np.concatenate((VR_img, Data_img), axis=1)
#cv2.imwrite('C:/Users/Taran/Desktop/' + 'VR_Data.png', VR_Data)




Diff_img = np.subtract(X_affine[-1], X_affine[0])
Diff_img = cv2.normalize(Diff_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#cv2.namedWindow('Difference', cv2.WINDOW_KEEPRATIO)
#cv2.imshow("Difference", Diff_img)


#PLOTTING EXPERIMENT
"""
fig = plt.figure(figsize=(8, 4))
plt.plot(range(len(lung_height_sorted)), lung_height_sorted, 'or', linewidth=0.5, markersize=1, zorder=0)
plt.xlabel('Slice number')
plt.ylabel('Lung height')
#fig.legend()
#plt.show()  
"""


##############PERFUSION##################

def hipass(img, sigma):
    return img - cv2.GaussianBlur(img, (3,3), sigma) + X.mean()


#perfusion_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.float32)
perfusion_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
#perfusion_mask = cv2.rectangle(perfusion_mask, (L_rect_sorted[0][0], L_rect_sorted[0][1]), (L_rect_sorted[0][0] + (R_rect_sorted[0][0]+R_rect_sorted[0][2] - L_rect_sorted[0][0]), L_rect_sorted[0][1] + L_rect_sorted[0][3]), 255, -1)
#perfusion_mask = cv2.rectangle(perfusion_mask, (L_rect_sorted[0][0]+L_rect_sorted[0][2], L_rect_sorted[0][1]), (R_rect_sorted[0][0],L_rect_sorted[0][1]+L_rect_sorted[0][3]), 255, -1)
#perfusion_mask = cv2.bitwise_or(perfusion_mask, end_mask)

#perfusion_mask = end_mask.copy()
perfusion_mask = cv2.line(perfusion_mask, (L_rect_sorted[0][0]+L_rect_sorted[0][2], L_rect_sorted[0][1]), (R_rect_sorted[0][0],R_rect_sorted[0][1]), 255, 1)
perfusion_mask = cv2.line(perfusion_mask, (R_rect_sorted[0][0], R_rect_sorted[0][1]), (R_rect_sorted[0][0]+R_rect_sorted[0][2],L_rect_sorted[0][1]+L_rect_sorted[0][3]), 255, 1)
perfusion_mask = cv2.line(perfusion_mask, (R_rect_sorted[0][0]+R_rect_sorted[0][2],L_rect_sorted[0][1]+L_rect_sorted[0][3]), (L_rect_sorted[0][0],L_rect_sorted[0][1]+L_rect_sorted[0][3]), 255, 1)
perfusion_mask = cv2.line(perfusion_mask, (L_rect_sorted[0][0],L_rect_sorted[0][1]+L_rect_sorted[0][3]), (L_rect_sorted[0][0]+L_rect_sorted[0][2], L_rect_sorted[0][1]), 255, 1)
cv2.floodFill(perfusion_mask, None, (IMG_HEIGHT//2,IMG_WIDTH//2), 255);
perfusion_mask = cv2.bitwise_or(perfusion_mask, end_mask)
perfusion_mask = cv2.rectangle(perfusion_mask, (0, 80), (128, 128), 0, -1)

#contours, hierarchy = cv2.findContours(end_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#perfusion_mask = cv2.drawContours(perfusion_mask, contours, -1, 255, 1)
#print(contours)
perfusion_mask = cv2.normalize(perfusion_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.namedWindow('perfusion_mask', cv2.WINDOW_KEEPRATIO)
cv2.imshow("perfusion_mask", perfusion_mask)

print('\nCompressing images...')



###############Другая последовательность изображений, L_rect_for_perf == L_rect здесь не работает. Надо сжать заново
#####WORK IN PROGRESS BEGIN###################

#sort_mode = 1

L_rect_for_perf = []
R_rect_for_perf = []
for i in tqdm(range(len(Y_for_perf))):
    cnt, hierarchy = cv2.findContours(Y_for_perf[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lung_x_0, lung_y_0, lung_w_0, lung_h_0 = cv2.boundingRect(cnt[0])
    lung_x_1, lung_y_1, lung_w_1, lung_h_1 = cv2.boundingRect(cnt[1])
    if(lung_x_0 > lung_x_1):
        R_rect_for_perf.append([lung_x_0, lung_y_0, lung_w_0, lung_h_0])
        L_rect_for_perf.append([lung_x_1, lung_y_1, lung_w_1, lung_h_1])
    else:
        R_rect_for_perf.append([lung_x_1, lung_y_1, lung_w_1, lung_h_1])
        L_rect_for_perf.append([lung_x_0, lung_y_0, lung_w_0, lung_h_0])
    #Y[i] = cv2.rectangle(Y[i], (L_rect_for_perf[i][0], L_rect_for_perf[i][1]), (L_rect_for_perf[i][0] + L_rect_for_perf[i][2], L_rect_for_perf[i][1] + L_rect_for_perf[i][3]), 100, -1)
    #Y[i] = cv2.rectangle(Y[i], (R_rect_for_perf[i][0], R_rect_for_perf[i][1]), (R_rect_for_perf[i][0] + R_rect_for_perf[i][2], R_rect_for_perf[i][1] + R_rect_for_perf[i][3]), 200, -1)
    Rect_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    Rect_img = cv2.rectangle(Rect_img, (L_rect_for_perf[i][0], L_rect_for_perf[i][1]), (L_rect_for_perf[i][0] + L_rect_for_perf[i][2], L_rect_for_perf[i][1] + L_rect_for_perf[i][3]), 100, -1)
    Rect_img = cv2.rectangle(Rect_img, (R_rect_for_perf[i][0], R_rect_for_perf[i][1]), (R_rect_for_perf[i][0] + R_rect_for_perf[i][2], R_rect_for_perf[i][1] + R_rect_for_perf[i][3]), 200, -1)
    #Y[i] = cv2.addWeighted(Y[i], 1, Rect_img, 0.5, 0)
    

sort_index_lung_height = np.argsort(lung_height_for_perf)#???????
min_lung_height_index = sort_index_lung_height[0]
points_ex_for_perf = [L_rect_for_perf[min_lung_height_index][0], L_rect_for_perf[min_lung_height_index][1]], [L_rect_for_perf[min_lung_height_index][0]+L_rect_for_perf[min_lung_height_index][2], L_rect_for_perf[min_lung_height_index][1]], [L_rect_for_perf[min_lung_height_index][0], L_rect_for_perf[min_lung_height_index][1]+L_rect_for_perf[min_lung_height_index][3]]
points_ex_for_perf = points_ex
#points_ex_for_perf = [[0,0], [60,0], [0,60]] #test
#####WORK IN PROGRESS END###################



#print(points_ex)
X_affine_for_perf = []
Y_affine_for_perf = []
for i in tqdm(range(len(X_for_perf))):
    points = [L_rect_for_perf[0][0], L_rect_for_perf[i][1]], [L_rect_for_perf[0][0]+L_rect_for_perf[0][2], L_rect_for_perf[i][1]], [L_rect_for_perf[0][0], L_rect_for_perf[i][1]+L_rect_for_perf[i][3]]
    affine_X_img = Affine_transform(X_for_perf[i], points, points_ex_for_perf)
    affine_Y_img = Affine_transform(Y_for_perf[i], points, points_ex_for_perf)
    X_affine_for_perf.append(affine_X_img)
    Y_affine_for_perf.append(affine_Y_img)

"""
#Saving compressed data
for i in tqdm(range(len(X_affine_for_perf))):    
    #cv2.imwrite('C:/Users/Taran/Desktop/PREFUL_1200/olyafirst1200_compressed/img_' + str(i) + '.tif', X_affine_for_perf[i])
    #cv2.imwrite('C:/Users/Taran/Desktop/PREFUL_1200/olyasecond1200_compressed/img_' + str(i) + '.tif', X_affine_for_perf[i])
    #cv2.imwrite('C:/Users/Taran/Desktop/PREFUL_1200/olyathird1200_compressed/img_' + str(i) + '.tif', X_affine_for_perf[i])
    cv2.imwrite('C:/Users/Taran/Desktop/Ready_data/OP_15/Compressed/img_' + str(i) + '.tif', X_affine_for_perf[i])
"""

X_affine_for_perf = np.asarray(X_affine_for_perf)
#X_hipass = hipass(X_affine_for_perf, 3)
X_hipass = X_affine_for_perf.copy()
print('len(X_hipass) = ', len(X_hipass))


#X_mean = sum(X_for_perf)/len(X_for_perf)
X_mean = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
X_sd = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
for i in range(len(X_for_perf)):
    X_mean += X_hipass[i]/len(X_hipass)
X_mean_8bit = cv2.normalize(X_mean, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.namedWindow('X_mean', cv2.WINDOW_KEEPRATIO)
cv2.imshow("X_mean", X_mean_8bit)
for i in range(len(X_for_perf)):
    #X_sd += (X_hipass[i] - X_mean)**2
    X_sd += (cv2.bitwise_or(X_hipass[i], X_hipass[i], mask = perfusion_mask) - cv2.bitwise_or(X_mean, X_mean, mask = perfusion_mask))**2
X_sd = np.sqrt(X_sd)
X_sd_8bit = cv2.normalize(X_sd, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.namedWindow('X_sd', cv2.WINDOW_KEEPRATIO)
cv2.imshow("X_sd", X_sd_8bit)

X_sd_th = X_sd.copy()
X_sd_th = cv2.bitwise_or(X_sd_th, X_sd_th, mask = perfusion_mask)
X_sd_th = (X_sd_th > 0.5*X_sd.max()).astype(np.uint8)
#X_sd_th = (X_sd_th > 0.9*X_sd_th.max()).astype(np.uint8)
kernel = np.ones((3, 5), np.uint8)
X_sd_th = cv2.erode(X_sd_th, kernel, 2)
"""
X_sd_th = np.zeros((128,128), dtype=np.uint8)
X_sd_th = cv2.rectangle(X_sd_th, (86, 76), (101, 97), 1, -1)
"""

def Mouse_draw(event, x, y, flags, param):
    #global ix, iy, drawing, img
    #global th1
    global ix, iy, drawing, X_mean_8bit
    global X_sd_th
    #drawing = 0
    point = (x, y)
    #radius = cv2.getTrackbarPos('Radius', 'TrackBar2')
    radius = 1
      
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = 1
        ix = x
        iy = y            
        #th1 = cv2.rectangle(th1, point,point,255,-1)
        X_sd_th = cv2.circle(X_sd_th, point, radius,255,-1)
        
    if event == cv2.EVENT_RBUTTONDOWN:
        drawing = 2
        ix = x
        iy = y            
        #th1 = cv2.rectangle(th1, point,point,0,-1)
        X_sd_th = cv2.circle(X_sd_th, point, radius,0,-1)
              
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == 1:
            #th1 = cv2.rectangle(th1, point,point,255,-1)
            X_sd_th = cv2.circle(X_sd_th, point, radius,255,-1)
        if drawing == 2:
            #th1 = cv2.rectangle(th1, point,point,0,-1)
            X_sd_th = cv2.circle(X_sd_th, point, radius,0,-1)
        cv2.imshow('X_sd_th', X_sd_th)
      
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = 0
        
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = 0
    cv2.imshow('X_sd_th', X_sd_th)
    X_sd_th_color = X_sd_th.copy()
    X_sd_th_color = cv2.cvtColor(X_sd_th_color, cv2.COLOR_GRAY2RGB)
    
    X_sd_th_color[..., 0] = 0
    X_sd_th_color[..., 1] = 0       
    
    X_mean_8bit = X_mean.copy()
    X_mean_8bit = cv2.normalize(X_mean_8bit, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    X_mean_8bit = cv2.cvtColor(X_mean_8bit, cv2.COLOR_GRAY2RGB)
    superimposed = cv2.addWeighted(X_mean_8bit, 1, X_sd_th_color, 0.5, 0)
    
    #cv2.namedWindow('Img+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("X_mean_ROI_select", superimposed)
    #cv2.imshow("Img", img_8bit)
        

if args.Q_ROI:  
    drawing = 0 
    X_sd_th = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
    cv2.namedWindow('X_mean_ROI_select', cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("X_mean_ROI_select", X_mean_8bit)
    cv2.setMouseCallback("X_mean_ROI_select", Mouse_draw)
    cv2.imshow('X_sd_th', X_sd_th)
    cv2.moveWindow('X_sd_th', 200, 800)
    cv2.resizeWindow('X_mean_ROI_select', 600, 600)
    cv2.waitKey()
    #exit()

    """
    roi = cv2.selectROI("X_mean", X_mean_8bit)
    X_sd_th = np.zeros((128,128), dtype=np.uint8)
    X_sd_th = cv2.rectangle(X_sd_th, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), 1, -1)
    """

#X_sd_th = cv2.bitwise_or(X_sd_th, X_sd_th, mask = perfusion_mask)
X_sd_th_8bit = cv2.normalize(X_sd_th, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.namedWindow('X_sd_th', cv2.WINDOW_KEEPRATIO)
cv2.imshow("X_sd_th", X_sd_th_8bit)

X_perf_masked = []
perf_signal = []
for i in range(len(X_for_perf)):
    #img = X_hipass[i]
    img = cv2.bitwise_or(X_hipass[i], X_hipass[i], mask = X_sd_th)
    X_perf_masked.append(img)
    perf_signal.append(img.mean())


perf_signal_AC = perf_signal.copy()
perf_signal_AC = np.asarray(perf_signal_AC)
perf_signal_AC = perf_signal_AC - perf_signal_AC.mean()

guess_x2 = [60, 1, 0, 60, 60, 0, 0]
guess = [60, 1, 0, 0]
"""
cut = 1500
X_indeces_for_perf = X_indeces_for_perf[:cut]
perf_signal = perf_signal[:cut]
X_phase_sorted = X_phase_sorted[:cut]
"""
"""
#sos = signal.butter(1, 0.9, 'hp', output='sos')
#perf_signal_hipass = signal.sosfiltfilt(sos, perf_signal)
perf_signal_fft = np.fft.fft(perf_signal_AC, axis=-1)
#perf_signal_fft[0:140] = 0
perf_signal_lopass = np.fft.ifft(perf_signal_fft, axis=-1)
perf_signal_hipass = np.fft.ifft(perf_signal_fft, axis=-1)
#perf_signal_hipass = perf_signal_AC - perf_signal_lopass*2
fig_perf_signal_fft = plt.figure(figsize=(14, 2), num='Perfusion signal fft')
plt.plot(X_indeces_for_perf, perf_signal_fft, 'o-', linewidth=0.5, label='data', markersize=2, color='tab:blue', alpha=0.5)
"""

print('perf_signal = ', perf_signal)


def line_filter_func(n, c):
    c_n = int(c*n/2)
    y = np.ones(n, dtype=float)
    for i in range(0, c_n):
        y[i]=i/c_n
    for i in range(0, c_n):
        y[n-1-i]=i/c_n
    #print(y)
    return y


def fft_hipass(y, c, o):
    y_fft = np.fft.fft(y)
    #n_to_delete = int(c*len(y_fft)/2)
    """
    y_fft[:n_to_delete] = 0
    y_fft[len(y_fft) - 1 - n_to_delete:] = 0
    """
    for i in range(o):
        y_fft = y_fft*line_filter_func(len(y_fft), c)
    y_ifft = np.fft.ifft(y_fft)
    fig_fft = plt.figure(figsize=(14, 2), num='Perfusion signal FFT')
    ax_fft = plt.subplot(1,1,1, label='FFT')
    ax_fft.plot(y_fft.real, color='tab:blue')
    ax_fft.plot(y_fft.imag, color='tab:orange')
    return y_ifft.real
    
perf_signal_hipass = fft_hipass(perf_signal, 0.5, 2)
print('perf_signal_hipass = ', perf_signal_hipass)

#parameters, covariance = curve_fit(cos_func_x2, X_indeces_for_perf, perf_signal_AC, p0=guess_x2, maxfev=50000)
#parameters, covariance = curve_fit(cos_func, X_indeces_for_perf, perf_signal_hipass, p0=guess, maxfev=50000)
parameters, covariance = curve_fit(cos_func_x2, X_indeces_for_perf, perf_signal_hipass, p0=guess_x2, maxfev=50000)
x_for_plotting_fit_perf = np.arange(0, len(perf_signal_AC), 0.1)
fit_cosine_perf = cos_func_x2(x_for_plotting_fit_perf, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6])
#fit_cosine_perf = cos_func(x_for_plotting_fit_perf, parameters[0], parameters[1], parameters[2], parameters[3])



print('X_indeces_for_perf: ', X_indeces_for_perf)
fig3 = plt.figure(figsize=(14, 2), num='Perfusion signal')
#ax = plt.subplot()
ax3 = plt.subplot(4, 1, (1, 3), label='lung_height')
plt.plot(X_indeces_for_perf, perf_signal_AC, 'o-', linewidth=0.5, label='data', markersize=2, color='tab:blue', alpha=0.5)
plt.plot(X_indeces_for_perf, perf_signal_hipass, 'o-', linewidth=0.5, label='data', markersize=2, color='tab:green', alpha=0.5)
plt.plot(x_for_plotting_fit_perf, fit_cosine_perf, '-', linewidth=0.5, label='data', markersize=2, color='tab:orange', alpha=0.5)

ax_phase3 = plt.subplot(4, 1, 4, label='Exhale', sharex=ax3)
ax_phase3.set_ylabel('Phase')
#plt.plot(X_indeces_for_perf, X_phase_sorted, 'o-', linewidth=0.5, label='data', markersize=2, color='tab:blue', alpha=0.5)

plt.subplots_adjust(left=0.05, right=0.99, bottom=0.13)








def Segment_fit (fit_X, fit_Y, func, guess, chi_sqr_critical):

    fig_signal_fit = plt.figure(figsize=(14, 2), num='Perfusion signal fit')
    ax_fit = plt.subplot(4, 1, (1, 3), label='fit_Y')
    plt.plot(X_indeces_for_perf, perf_signal_AC, 'o-', linewidth=0.5, label='data', markersize=2, color='tab:blue', alpha=0.5)
    x_from = 0
    x_to = x_from + len(guess)
    chi_sqr_flag = 0
    #chi_sqr_critical = 0.4
    X_phase = []
    segment_divider = []
    plotting_phase = []
    X_freeq = []
    X_C = []
    TypeError_flag = 0


    while chi_sqr_flag == 0:
        x_range = fit_X[x_from:x_to+1]
        y_range = fit_Y[x_from:x_to+1]
        with shutup.mute_warnings:
            try:
                parameters, covariance = curve_fit(cos_func, x_range, y_range, p0=guess, maxfev=5000000, bounds=([-10, 1, -np.inf, -np.inf], [10, 10, np.inf, np.inf]))            
            except TypeError:
                parameters = [0,0,0,0]
                TypeError_flag = 1
        fit_cosine = cos_func(x_range, parameters[0], parameters[1], parameters[2], parameters[3])
        chi_sqr = 0
        for i in range(0, len(x_range)):
            chi_sqr += abs(((fit_Y[i+x_from] - fit_cosine[i])**2)/(fit_cosine[i]*len(x_range)))
            #print('     fit_cosine[i]*len(x_range) = ', fit_cosine[i]*len(x_range))    
        #print('chi_sqr = ', chi_sqr)
        if TypeError_flag == 1:
            chi_sqr = 100
        if chi_sqr > chi_sqr_critical and TypeError_flag != 1:
            #chi_sqr_flag = 1
            x_to -= 1
            x_range = fit_X[x_from:x_to+1]
            y_range = fit_Y[x_from:x_to+1]
            with shutup.mute_warnings:
                parameters, covariance = curve_fit(cos_func, x_range, y_range, p0=guess, maxfev=50000, bounds=([-10, 1, -np.inf, -np.inf], [10, 10, np.inf, np.inf]))
            parameters[2] = parameters[2]%(2*np.pi)
            ######################CHI_SQR
            chi_sqr = 0        
            for i in range(0, len(x_range)):
                chi_sqr += abs(((fit_Y[i+x_from] - fit_cosine[i])**2)/(fit_cosine[i]*len(x_range)))
            print("Segment result: [", fit_X[x_from], fit_X[x_to], ']', 'len =', x_to-x_from, 'chi_sqr =', round(chi_sqr, 3), 'parameters = ', round(parameters[0], 3), round(parameters[1], 3), round(parameters[2], 3), round(parameters[3], 3))
            ########################PHASE
            for i in range(0, len(x_range)-1):
                #      y = D*np.cos(E*x + F) + C
                #phase = math.acos((fit_cosine[i]-parameters[3])/parameters[0])
                #phase = (x_range[i]-parameters[2])*parameters[1] - ((x_range[i]-parameters[2])*parameters[1])//(2*math.pi)*2*math.pi  
                #phase = x_range[i]*parameters[1] + parameters[2] - ((x_range[i]*parameters[1] + parameters[2])//(2*math.pi))*2*math.pi  #Gold
                if parameters[0] < 0:
                    ps = np.pi
                else:
                    ps = 0
                phase = (x_range[i]*parameters[1] + parameters[2] + ps)%(2*np.pi)
                #phase = x_range[i]*parameters[1] + parameters[2]
                X_phase.append(phase)
                X_C.append(parameters[3])
                X_freeq.append(parameters[1])
            #############################    
            segment_divider.append(fit_X[x_from])
            x_for_plotting_fit = np.arange(x_range[0], x_range[-1], 0.1)
            fit_cosine = func(x_for_plotting_fit, parameters[0], parameters[1], parameters[2], parameters[3])
            plt.plot(x_for_plotting_fit, fit_cosine, '-', linewidth=1, color='tab:orange') # 00FFEA 22FF00
            plt.plot([x_range[-1], x_range[-1]], [fit_Y.min(), fit_Y.max()], '-', linewidth=0.5, alpha=0.7, color='#0D2249')
            for i in range(0, len(x_for_plotting_fit)):
                #      y = D*np.cos(E*x + F) + C
                #plotting_phase.append(x_for_plotting_fit[i]*parameters[1] + parameters[2] - ((x_for_plotting_fit[i]*parameters[1] + parameters[2])//(2*math.pi))*2*math.pi)
                if parameters[0] < 0:
                    ps = np.pi
                else:
                    ps = 0
                plotting_phase.append((x_for_plotting_fit[i]*parameters[1] + parameters[2] + ps)%(2*np.pi))
            x_from = x_to
            x_to = x_from + len(guess)
            #plt.plot(x_range, fit_cosine, '-g', linewidth=1, label='fit')
            
        else:
            x_to += 1
            if x_to >= len(fit_X):
                """
                if x_to - x_from < 4:
                    segment_divider.append(fit_X[x_from])
                    break
                """
                x_to == len(fit_X)-1
                chi_sqr_flag = 1
                x_range = fit_X[x_from:x_to+1]
                y_range = fit_Y[x_from:x_to+1]
                with shutup.mute_warnings:
                    try:
                        parameters, covariance = curve_fit(cos_func, x_range, y_range, p0=guess, maxfev=50000, bounds=([-10, 1, -np.inf, -np.inf], [10, 10, np.inf, np.inf]))
                    except TypeError:
                        parameters = guess
                ######################CHI_SQR
                chi_sqr = 0        
                for i in range(0, len(x_range)):
                    chi_sqr += ((fit_Y[i+x_from] - fit_cosine[i])**2)/(fit_cosine[i]*len(x_range))
                #print("Segment result", x_from, x_to, chi_sqr)
                ########################PHASE
                for i in range(0, len(x_range)):
                    #phase = x_range[i]*parameters[1] + parameters[2] - ((x_range[i]*parameters[1] + parameters[2])//(2*math.pi))*2*math.pi  
                    if parameters[0] < 0:
                        ps = np.pi
                    else:
                        ps = 0
                    phase = (x_range[i]*parameters[1] + parameters[2] + ps)%(2*np.pi)
                    X_phase.append(phase)
                    X_C.append(parameters[3])
                    X_freeq.append(parameters[1])
                segment_divider.append(fit_X[x_from])
                x_for_plotting_fit = np.arange(x_range[0], x_range[-1], 0.1)
                fit_cosine = cos_func(x_for_plotting_fit, parameters[0], parameters[1], parameters[2], parameters[3])
                plt.plot(x_for_plotting_fit, fit_cosine, '-', linewidth=1, color='tab:orange') # 00FFEA 22FF00
                for i in range(0, len(x_for_plotting_fit)):
                    #      y = D*np.cos(E*x + F) + C
                    #plotting_phase.append(x_for_plotting_fit[i]*parameters[1] + parameters[2] - ((x_for_plotting_fit[i]*parameters[1] + parameters[2])//(2*math.pi))*2*math.pi)
                    if parameters[0] < 0:
                        ps = np.pi
                    else:
                        ps = 0
                    plotting_phase.append((x_for_plotting_fit[i]*parameters[1] + parameters[2] + ps)%(2*np.pi))
                segment_divider.append(fit_X[-1])


    
    
    print('segment_divider', segment_divider)

    bad_segments = []
    bad_slices = []
    for i in range(0, len(segment_divider)-1):
        X_C_index_arr = np.where(fit_X == segment_divider[i])
        X_C_index = X_C_index_arr[0][0]
        #print(X_C_index)
        #if segment_divider[i+1] - segment_divider[i] < 6 or X_C[X_C_index]-fit_Y.mean() > 0.1*(fit_Y.max()-fit_Y.min()): #change .mean() to parameters[3]
        if segment_divider[i+1] - segment_divider[i] < len(guess):
            print(i)
            bad_segments.append(i)
            for j in range(len(Y)):
                if fit_X[j] < segment_divider[i+1] and fit_X[j] >= segment_divider[i]:
                    bad_slices.append(j)
    for i in range(len(bad_segments)):
        ax_fit.axvspan(segment_divider[bad_segments[i]], segment_divider[bad_segments[i]+1], facecolor='0.42', alpha=0.5)
    print('bad_slices', bad_slices)
    print('bad_indeces', fit_X[bad_slices])

    """
    fit_X = np.delete(fit_X, bad_slices, 0)
    X = np.delete(X, bad_slices, 0)
    Y = np.delete(Y, bad_slices, 0)
    L_rect = np.delete(L_rect, bad_slices, 0)
    R_rect = np.delete(R_rect, bad_slices, 0)
    fit_Y = np.delete(fit_Y, bad_slices, 0)
    X_phase = np.delete(X_phase, bad_slices, 0)
    X_C = np.delete(X_C, bad_slices, 0)
    print('bad segments found: ', len(bad_slices))
    """
    #plt.plot(fit_X, X_C, 'o-', linewidth=0.5, markersize=1, color='gold', zorder=1) #Plot C


    ax_phase = plt.subplot(4, 1, 4, label='Exhale', sharex=ax_fit)
    ax_phase.set_ylabel('Phase')
    ax_phase.set_ylabel('Фаза')
    ax_phase.set_xlabel('Номер скана')
    ax_phase.yaxis.set_minor_locator(ticker.MultipleLocator(np.pi/4))
    ax_phase.yaxis.set_major_locator(ticker.MultipleLocator(np.pi))
    ax_phase.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax_phase.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax_phase.set_axisbelow(True)
    ax_phase.yaxis.grid(color='0.95', which='major', zorder=0)
    ax_phase.xaxis.grid(color='0.95', which='both', zorder=0)
    ax_phase.set_yticks([0, np.pi, 2*np.pi])
    ax_phase.set_yticklabels(['0', 'π', '2π'])
    for i in range(len(bad_segments)):
        ax_phase.axvspan(segment_divider[bad_segments[i]], segment_divider[bad_segments[i]+1], facecolor='0.42', alpha=0.5)

    plt.plot(fit_X, X_phase, 'o', linewidth=0.5, label='data', markersize=2, color='tab:blue', alpha=0.5)
    x_for_plotting_phase = np.arange(0, len(plotting_phase)/10, 0.1)
    plt.plot(x_for_plotting_phase, plotting_phase, '-', linewidth=0.5, label='data', markersize=2, color='tab:green', alpha=0.5)
    for i in range(len(segment_divider)):
        plt.plot([segment_divider[i], segment_divider[i]], [0, 2*math.pi], '-', linewidth=0.5, alpha=0.7, color='#0D2249')
    #ax_phase.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))

    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.13)
    """
    if matplotlib_stop == 1:
        plt.show()
    else:
        plt.show(block=False)
    """
    return X_phase, segment_divider, X_freeq, X_C, bad_segments, bad_slices

X_phase_for_perf, segment_divider_for_perf, X_freeq_for_perf, X_C_for_perf, bad_segments_for_perf, bad_slices_for_perf = Segment_fit(X_indeces_for_perf, perf_signal_hipass, cos_func, [1, 1, 0, 0], 0.2)


X_hipass = np.delete(X_hipass, bad_slices_for_perf, 0) #X
Y_affine_for_perf = np.delete(Y_affine_for_perf, bad_slices_for_perf, 0) #Y
X_indeces_for_perf = np.delete(X_indeces_for_perf, bad_slices_for_perf, 0)
perf_signal_hipass = np.delete(perf_signal_hipass, bad_slices_for_perf, 0)
X_phase_for_perf = np.delete(X_phase_for_perf, bad_slices_for_perf, 0)
X_freeq_for_perf = np.delete(X_freeq_for_perf, bad_slices_for_perf, 0)
X_C_for_perf = np.delete(X_C_for_perf, bad_slices_for_perf, 0)

#print('X_freeq_for_perf = ', X_freeq_for_perf)

X_freeq_for_perf_mean = X_freeq_for_perf.mean()
print('X_freeq_for_perf_mean = ', X_freeq_for_perf_mean)


#sort_mode = 1
sort_index_for_perf = np.argsort(X_phase_for_perf)
X_hipass_sorted = []
Y_affine_for_perf_sorted = []
X_indeces_for_perf_sorted = []
perf_signal_sorted = []
X_phase_for_perf_sorted = []
#X_C_for_perf_sorted = []
for i in tqdm(range(len(sort_index_for_perf))):
    X_hipass_sorted.append(X_hipass[sort_index_for_perf[i]])
    Y_affine_for_perf_sorted.append(Y_affine_for_perf[sort_index_for_perf[i]])
    X_indeces_for_perf_sorted.append(X_indeces_for_perf[sort_index_for_perf[i]])
    perf_signal_sorted.append(perf_signal_hipass[sort_index_for_perf[i]])
    X_phase_for_perf_sorted.append(X_phase_for_perf[sort_index_for_perf[i]])
#print(lung_height_sorted)


print('len(perf_signal_sorted)', len(perf_signal_sorted))
print('len(X_phase_for_perf_sorted)', len(X_phase_for_perf_sorted))


new_indeces_for_perf = np.arange(len(X_hipass_sorted))
perf_signal_sorted = np.asarray(perf_signal_sorted)
perf_signal_sorted_guess = [perf_signal_sorted.max()-perf_signal_sorted.min(), 1/len(perf_signal_sorted), 0, perf_signal_sorted.mean()]
perf_signal_sorted_fit_parameters, perf_signal_sorted_fit_covariance = curve_fit(cos_func, new_indeces_for_perf, perf_signal_sorted, p0=perf_signal_sorted_guess, maxfev=50000, bounds=([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
perf_signal_sorted_fit = cos_func(new_indeces_for_perf, perf_signal_sorted_fit_parameters[0], perf_signal_sorted_fit_parameters[1], perf_signal_sorted_fit_parameters[2], perf_signal_sorted_fit_parameters[3])

print('Finding samples with bad phase...')
perf_signal_sorted_bad_indeces = []
for i in tqdm(range(len(perf_signal_sorted))):
    #print(i)
    if abs(perf_signal_sorted[i] - perf_signal_sorted_fit[i]) > (perf_signal_sorted.max()-perf_signal_sorted.min())*0.2:
    #if perf_signal_sorted[i] - sorted_fit[i] > 7:
        perf_signal_sorted_bad_indeces.append(i)
print('perf_signal_sorted_bad_indeces found: ', len(perf_signal_sorted_bad_indeces))

fig_perf_signal_sorted = plt.figure(figsize=(14, 2), num='Perfusion signal sorted')
#ax = plt.subplot()
ax_perf_signal_sorted = plt.subplot(4, 1, (1, 3), label='perf_signal')
plt.plot(new_indeces_for_perf, perf_signal_sorted, 'o-', linewidth=0.5, label='data', markersize=2, color='tab:green', alpha=0.5)
plt.plot(new_indeces_for_perf, perf_signal_sorted_fit, '-', linewidth=0.5, label='data', markersize=2, color='tab:orange', alpha=0.5)
for i in range(len(perf_signal_sorted_bad_indeces)):    
    plt.plot(new_indeces_for_perf[perf_signal_sorted_bad_indeces[i]], perf_signal_sorted[perf_signal_sorted_bad_indeces[i]], 'o', linewidth=2, label='data', markersize=2, color='tab:red', alpha=1)

ax_phase_perf_signal_sorted = plt.subplot(4, 1, 4, label='Exhale', sharex=ax_perf_signal_sorted)
ax_phase_perf_signal_sorted.set_ylabel('Phase')
plt.plot(new_indeces_for_perf, X_phase_for_perf_sorted, 'o-', linewidth=0.5, label='data', markersize=2, color='tab:blue', alpha=0.5)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.13)

new_indeces_for_perf = np.delete(new_indeces_for_perf, perf_signal_sorted_bad_indeces, 0)
X_hipass_sorted = np.delete(X_hipass_sorted, perf_signal_sorted_bad_indeces, 0) #X
Y_affine_for_perf_sorted = np.delete(Y_affine_for_perf_sorted, perf_signal_sorted_bad_indeces, 0) #Y
X_indeces_for_perf_sorted = np.delete(X_indeces_for_perf_sorted, perf_signal_sorted_bad_indeces, 0)
perf_signal_sorted = np.delete(perf_signal_sorted, perf_signal_sorted_bad_indeces, 0)
X_phase_for_perf_sorted = np.delete(X_phase_for_perf_sorted, perf_signal_sorted_bad_indeces, 0)




min_lung_height_index = sort_index_lung_height[0]

Sblood = perf_signal_sorted[0]


#############WORK IN PROGRESS BEGIN##################
T_slices = 0.0044*64#seconds
T_slices = 0.0044*80#seconds
T_slices = 0.0045*100#seconds
T_slices = 0.0046*126#seconds
tblood = 2*np.pi/X_freeq_for_perf_mean * T_slices
print('tblood =', tblood, 'seconds; fblood =', 1/tblood, 'Hz =', 60/tblood, 'bpm')

def Create_Qquant(Q, Sblood, tblood):
    Qquant = Q/Sblood * 1/(2*tblood)
    return Qquant

def Create_Perf_map(Full_img, Empty_img):
    Qmap = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    Qmap = Full_img - Empty_img
    
    count_positive = 0
    count_negative = 0
    Qmap_mean = 0
    for i in range(0, len(Qmap)):
        for j in range(0, len(Qmap)):
            if Qmap[i][j]>0:
                count_positive+=1
                Qmap_mean+=Qmap[i][j]
            if Qmap[i][j]<0:
                count_negative+=1
                #Qmap[i][j] = 0
    #count_coefficient = (count_positive-count_negative)/(2*(count_positive+count_negative)) + 0.5
    #Qmap_mean = np.mean(Qmap)
    Qmap_mean=Qmap_mean/count_positive
    #STANDART DEVIATION
    sd = 0
    for i in range(0, len(Qmap)):
        for j in range(0, len(Qmap)):
            if Qmap[i][j]>0:
                sd += (Qmap[i][j] - Qmap_mean)**2
                #Qmap[i][j] = 0
    sd=math.sqrt(sd/count_positive)
    #print('\nQmap:', count_positive, '/', count_negative, '  ', count_coefficient)
    max = Qmap.max()
    min = Qmap.min()
    print('\nQmap:', count_positive, '/', count_negative)
    print('Mean:', Qmap_mean, 'sd:', sd)
    return Qmap, count_positive, count_negative, Qmap_mean, sd, min, max


X_masked_for_perf = []
for i in tqdm(range(len(X_hipass_sorted))):
    img = cv2.bitwise_or(X_hipass_sorted[i], X_hipass_sorted[i], mask=end_mask)
    X_masked_for_perf.append(img)



def Create_interpolation(array):
    mid_point = len(array)//2
    quarter_point = len(array)//4
    sum_c = 0
    n = len(array)//50
    if n % 2 == 0:
        n += 1
    img_min = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    img_max = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    img_mid = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    for i in range(-(n//2), (n//2)+1):
        c = (n/2-abs(i))/(n/2)
        sum_c += c
        img_min += array[mid_point + i] * c
        img_max += array[-1 + i] * c
        img_mid += array[quarter_point + i] * c        
    img_min /= sum_c
    img_max /= sum_c
    img_mid /= sum_c
    return img_min, img_max, img_mid

def Mask(img, mask):
    return cv2.bitwise_or(img, img, mask=mask)

img_min_perf, img_max_perf, img_mid_perf = Create_interpolation(X_masked_for_perf)



Qquant_max = Create_Qquant(img_max_perf, Sblood, tblood)
Qquant_min = Create_Qquant(img_min_perf, Sblood, tblood)


Qquant_max = Mask(Qquant_max, end_mask)
Qquant_min = Mask(Qquant_min, end_mask)



Qmap, Qmap_pos, Qmap_neg, Qmap_mean, Qmap_sd, Qmap_min, Qmap_max = Create_Perf_map(Qquant_max, Qquant_min)
Qmap_8bit = Qmap.copy()
Qmap_8bit = cv2.normalize(Qmap_8bit, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
Qmap_8bit = cv2.bitwise_or(Qmap_8bit, Qmap_8bit, mask=end_mask)
cv2.namedWindow('Qmap', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Qmap", Qmap_8bit)




data_x = 62
data_x_step = 15

data_y = 25
data_y_step = 15

QData_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
QData_img = cv2.rectangle(QData_img,(8,19),(18,108),100,-1)
for i in range(0, 88):
    color = (88-i)*255//88
    QData_img = cv2.rectangle(QData_img,(9,i+20),(17,i+20),color,-1)

QData_img = cv2.putText(QData_img, str(round(Qmap_max, 3)), (22,data_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
QData_img = cv2.putText(QData_img, str(round(Qmap_min, 3)), (22,108), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
QData_img = cv2.putText(QData_img, 'pos: '+str(Qmap_pos), (data_x,data_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
QData_img = cv2.putText(QData_img, 'neg: '+str(Qmap_neg), (data_x,data_y+data_y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
QData_img = cv2.putText(QData_img, 'mean: '+str(round(Qmap_mean, 3)), (data_x,data_y+2*data_y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
QData_img = cv2.putText(QData_img, 'sd: '+str(round(Qmap_sd, 3)), (data_x,data_y+3*data_y_step), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)

QData_img = cv2.rectangle(QData_img,(data_x,data_y+3*data_y_step+6),(data_x+60,data_y+3*data_y_step+6),255,-1)
cv2.namedWindow('QData', cv2.WINDOW_KEEPRATIO)
cv2.imshow("QData", QData_img)

#############WORK IN PROGRESS END##################





























"""
if matplotlib_stop == 1:
    plt.show()
else:
    plt.show(block=False)
"""
plt.show(block=False)
plt.pause(0.001)
#input("hit[enter] to end.")
plt.waitforbuttonpress(0) # this will wait for indefinite time
plt.close('all') # all open plots are correctly closed after each run


current_img = 0
Show_images(current_img)





cv2.moveWindow('X', 1, 0)
cv2.moveWindow('Y', 400, 0)
cv2.moveWindow('Img+mask', 800, 0)
cv2.moveWindow('VR', 1200, 0)
cv2.moveWindow('Data', 1600, 0)

cv2.moveWindow('X_perf', 1, 400)
cv2.moveWindow('Y_perf', 400, 400)
cv2.moveWindow('perfusion_mask', 800, 400)
cv2.moveWindow('Qmap', 1200, 400)
cv2.moveWindow('QData', 1600, 400)

cv2.moveWindow('X_hipass', 1, 800)
cv2.moveWindow('X_sd_th', 400, 800)
cv2.moveWindow('X_perf_masked', 800, 800)

cv2.moveWindow('X_mean', 1, 1200)
cv2.moveWindow('X_sd', 400, 1200)




print(path)
path = dirname(path)
file_folder = basename(dirname(dirname(path))) + '/' + basename(dirname(path))
Data_img_top = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
Data_img_top = cv2.putText(Data_img_top, file_folder, (1,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
Data_img_top = cv2.putText(Data_img_top, 'VR', (116,70), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)

Data_img_bottom = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
Data_img_bottom = cv2.putText(Data_img_bottom, 'Qmap', (100,70), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)

Results_img_top = np.concatenate((Data_img_top, VR_img, Data_img), axis=1)
Results_img_bottom = np.concatenate((Data_img_bottom, Qmap_8bit, QData_img), axis=1)
Results_img = np.concatenate((Results_img_top, Results_img_bottom), axis=0)

cv2.namedWindow('Results_img')
cv2.imshow("Results_img", Results_img)
cv2.moveWindow('Results_img', 1200, 1200)

end_time = time.time()
print('Execution time:', end_time - start_time, 's')


img_len = len(X)
while(1):
    full_key_code = cv2.waitKeyEx(30)
    #print(full_key_code)
    if full_key_code == 65363:
        current_img += 1
        if current_img >= img_len:
            current_img = img_len-1
        Show_images(current_img)
        #print('image ' + str(current_img) + ' / ' + str(img_len))
        print('image ' + str(current_img) + ' / ' + str(img_len) + ' lung_height = ' + str(lung_height[current_img]))
    if full_key_code == 65361:
        current_img -= 1        
        if current_img < 0:
            current_img = 0
        Show_images(current_img)
        print('image ' + str(current_img) + ' / ' + str(img_len))
    """
    if full_key_code == 32:
        #a = 'VR/OP_15_1200/1000_'
        a = 'VR/OP_NS_1800/1200_mean_100'
        cv2.imwrite('C:/Users/Taran/Desktop/' + a + 'VR_Data.png', VR_Data)
        cv2.imwrite('C:/Users/Taran/Desktop/' + a + 'VR_img.png', VR_img)
        #cv2.imwrite('C:/Users/Taran/Desktop/' + a + 'Data_img.png', Data_img)
        print('VR_Data.png saved!')
    """
    if full_key_code == 27:
            exit()
    #plt.show(block=False)
