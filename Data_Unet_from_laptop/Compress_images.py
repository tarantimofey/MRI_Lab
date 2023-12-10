#C:/Users/Taran/AppData/Local/Programs/Python/Python39/python.exe -i "$(FULL_CURRENT_PATH)"

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage
from scipy import signal
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

interpolation = 1
sort_mode = 1

border_data_delition = 1

matplotlib_stop = 0


#path = 'C:/Users/Taran/Desktop/Ready_data/OP_SS/Png/Image/'
path = 'C:/Users/Taran/Desktop/Ready_data/OP_SS/Tiff/'
path = 'C:/Users/Taran/Desktop/Ready_data/OP_15/Tiff/'
path = 'C:/Users/Taran/Desktop/Ready_data/OP_NS/Tiff/'

#path = 'C:/Users/Taran/Desktop/Makur_SS/Tiff/'
#path = 'C:/Users/Taran/Desktop/arina487/Tiff/'
path = 'C:/Users/Taran/Desktop/OP_NS_1800/Tiff/'
#path = 'C:/Users/Taran/Desktop/PREFUL_1200/olyafirst1200/Tiff/Norm/'
#path = 'C:/Users/Taran/Desktop/PREFUL_1200/olyasecond1200/Tiff/Norm/'
path = 'C:/Users/Taran/Desktop/PREFUL_1200/olyathird1200/Tiff/Image/'

#model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/Lung/4Run/MRI_lung_1.h5')
model = tf.keras.models.load_model('C:/Users/Taran/Desktop/Data_Unet/Models/Lung/5Run/MRI_lung_1.h5')

#######################################################################################################
def Show_images(n):

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
    """
    X_img_hipass = X_hipass[n]
    X_img_hipass = X[n]
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
    """

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
    cv2.drawContours(th, [cnt[0], cnt[1]], -1, 255, -1)
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
    sd=sd/count_positive
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


print('\n\nReading images from ', path)
ids = listdir(path)
X = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
print('\nResizing images...')
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

print('\nPredicting masks...')
Y_pred = model.predict(X, verbose=1)
Y_pred = (Y_pred > 0.5).astype(np.uint8)

#print(type(X[0,0,0]))
#X = X[:, :, :, 0].astype(np.uint8)
#X = X[:, :, :, 0].astype(np.float32)
X = X.astype(np.float32)
#X = X[:, :, :, 0]
Y = Y_pred[:, :, :, 0].astype(np.uint8)

"""
for i in range(len(X)):
    X[i] = cv2.normalize(X[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
"""


print('\nRefining masks...')
for i in tqdm(range(len(Y_pred))):
    Y[i] = Refine_mask(Y[i])






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
    
    
print('\nDeleting bad data...')
bad_slices = []
L_rect = np.asarray(L_rect)
R_rect = np.asarray(R_rect)
lung_height = L_rect[:, 3]
X_indeces = np.arange(len(X))


fig = plt.figure(figsize=(14, 2))
#ax = plt.subplot()
ax = plt.subplot(4, 1, (1, 3), label='lung_height')

for i in tqdm(range(len(Y))):
    if L_rect[i][1]+L_rect[i][3] < IMG_HEIGHT/2:
        bad_slices.append(i)
#print('bad_slices', bad_slices)
X_indeces = np.delete(X_indeces, bad_slices, 0)
X = np.delete(X, bad_slices, 0)
Y = np.delete(Y, bad_slices, 0)
L_rect = np.delete(L_rect, bad_slices, 0)
R_rect = np.delete(R_rect, bad_slices, 0)
lung_height = np.delete(lung_height, bad_slices, 0)
print('bad slices found: ', len(bad_slices))

#Copying data for perfusion

X_indeces_for_perf = X_indeces.copy()
X_for_perf = X.copy()
Y_for_perf = Y.copy()
L_rect_for_perf = L_rect.copy()
R_rect_for_perf = R_rect.copy()
lung_height_for_perf = lung_height.copy()





print('\nSorting images...')
#print(L_rect)
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.set_axisbelow(True)
ax.xaxis.grid(color='0.95', which='both', zorder=0)
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


sort_mode = 0
sort_index_lung_height = np.argsort(lung_height)
#sort_index_phase = np.argsort(X_phase)
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
    #X_phase_sorted.append(X_phase[sort_index[i]])
#print(lung_height_sorted)
min_lung_height_index = sort_index_lung_height[0]





print('\nCompressing images...')
points_ex = [L_rect_sorted[min_lung_height_index][0], L_rect_sorted[min_lung_height_index][1]], [L_rect_sorted[min_lung_height_index][0]+L_rect_sorted[min_lung_height_index][2], L_rect_sorted[min_lung_height_index][1]], [L_rect_sorted[min_lung_height_index][0], L_rect_sorted[min_lung_height_index][1]+L_rect_sorted[min_lung_height_index][3]]

#print(points_ex)
X_affine = []
Y_affine = []
"""
for i in tqdm(range(len(X_sorted))):
    points = [L_rect_sorted[0][0], L_rect_sorted[i][1]], [L_rect_sorted[0][0]+L_rect_sorted[0][2], L_rect_sorted[i][1]], [L_rect_sorted[0][0], L_rect_sorted[i][1]+L_rect_sorted[i][3]]
    affine_X_img = Affine_transform(X_sorted[i], points, points_ex)
    affine_Y_img = Affine_transform(Y_sorted[i], points, points_ex)
    X_affine.append(affine_X_img)
    Y_affine.append(affine_Y_img)
"""
for i in tqdm(range(len(X_sorted))):
    points = [L_rect[0][0], L_rect[i][1]], [L_rect[0][0]+L_rect[0][2], L_rect[i][1]], [L_rect[0][0], L_rect[i][1]+L_rect[i][3]]
    affine_X_img = Affine_transform(X[i], points, points_ex)
    affine_Y_img = Affine_transform(Y[i], points, points_ex)
    X_affine.append(affine_X_img)
    Y_affine.append(affine_Y_img)



#Saving compressed data
for i in tqdm(range(len(X_affine))):    
    #cv2.imwrite('C:/Users/Taran/Desktop/PREFUL_1200/olyafirst1200/Tiff/Compressed/img_' + str(i) + '.tif', X_affine[i])
    #cv2.imwrite('C:/Users/Taran/Desktop/PREFUL_1200/olyasecond1200/Tiff/Compressed/img_' + str(i) + '.tif', X_affine[i])
    cv2.imwrite('C:/Users/Taran/Desktop/PREFUL_1200/olyathird1200/Tiff/Compressed/img_' + str(i) + '.tif', X_affine[i])










img_len = len(X_affine)
current_img = 0
Show_images(current_img)

while(1):
    full_key_code = cv2.waitKeyEx(0)
    if full_key_code == 2555904:
        current_img += 1
        if current_img >= img_len:
            current_img = img_len-1
        Show_images(current_img)
        #print('image ' + str(current_img) + ' / ' + str(img_len))
        print('image ' + str(current_img) + ' / ' + str(img_len) + ' lung_height = ' + str(lung_height[current_img]))
    if full_key_code == 2424832:
        current_img -= 1        
        if current_img < 0:
            current_img = 0
        Show_images(current_img)
        print('image ' + str(current_img) + ' / ' + str(img_len))
    if full_key_code == 32:
        """
        cv2.imwrite('C:/Users/Taran/Desktop/' + 'Img' + str(n) + '.png', superimposed)
        print('image ' + str(current_img) + ' saved!')
        """
        a = 'VR/OP_NS_'
        cv2.imwrite('C:/Users/Taran/Desktop/' + a + 'VR_Data.png', VR_Data)
        cv2.imwrite('C:/Users/Taran/Desktop/' + a + 'VR_img.png', VR_img)
        cv2.imwrite('C:/Users/Taran/Desktop/' + a + 'Data_img.png', Data_img)
        print('VR_Data.png saved!')
    #plt.show(block=False)