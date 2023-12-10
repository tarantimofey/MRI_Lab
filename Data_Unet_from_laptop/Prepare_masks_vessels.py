#C:\Users\Taran\AppData\Local\Programs\Python\Python39\python.exe -i "$(FULL_CURRENT_PATH)"
import cv2
import numpy as np
from scipy.optimize import curve_fit
import scipy.signal as signal
import matplotlib.pyplot as plt
import math 
import My_Preful_lib.My_Preful_lib as mp
from os import listdir
from os.path import isfile, join
import cv2
import imageio

import os
import win32api, win32con

from tqdm import tqdm


WINDOW_SIZE = 750


def Show_XY_image(n): 
    global XY_img 
    XY_img  = images[current_img] 
    XY_img = cv2.normalize(XY_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    #cv2.namedWindow('XY', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow("XY", XY_img)
    
    XY_img_color = XY_img.copy()
    XY_img_color = cv2.cvtColor(XY_img_color, cv2.COLOR_GRAY2RGB)
    
    XY_mask = mask[current_img] #Берём слайс из трёхмерного массива маски
    XY_mask = cv2.normalize(XY_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    XY_mask_color = XY_mask.copy()
    XY_mask_color = cv2.cvtColor(XY_mask_color, cv2.COLOR_GRAY2RGB)
    XY_mask_color[:,:,0] = 0
    XY_mask_color[:,:,1] = 0
    """
    XY_mask_color[XY_mask_color[:,:,0] == 1] = (0,0,255)
    XY_mask_color[XY_mask_color[:,:,0] == 2] = (0,255,0)
    XY_mask_color[XY_mask_color[:,:,0] == 3] = (255,0,0)
    """
    superimposed = cv2.addWeighted(XY_img_color, 1, XY_mask_color, Mask_alpha, 0) 
    cursor_img_color = cv2.cvtColor(cursor_img, cv2.COLOR_GRAY2RGB)
    superimposed = cv2.addWeighted(superimposed, 1, cursor_img_color, 0.5, 0) 
    
    #superimposed = XY_img_color
    cv2.namedWindow('XY+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("XY+mask", superimposed)   #Апдейтим картинку
    cv2.resizeWindow("XY+mask", WINDOW_SIZE, WINDOW_SIZE)
        

def Mouse_draw(event, x, y, flags, param):
    #win32api.SetCursor(None)
    win32api.SetCursor(win32api.LoadCursor(0, win32con.OCR_NORMAL))
    global ix, iy, drawing, cursor_img
    #drawing = 0
    point = (x, y)  
    radius = cv2.getTrackbarPos('Radius', 'TrackBar2') 
    drawing_color = 255
    
    #cursor_img = np.zeros((IMG_WIDTH,IMG_WIDTH), np.uint8)
    
    XY_mask = np.zeros((IMG_WIDTH,IMG_WIDTH), np.uint8)
    XY_mask = cv2.addWeighted(XY_mask, 1, mask[current_img], 1, 0) 
    
    if event == cv2.EVENT_LBUTTONDOWN: #Левой кнопкой мышки рисуем
        drawing = 1
        XY_mask = cv2.circle(XY_mask, point, radius,drawing_color,-1)
        mask[current_img] = XY_mask #Апдейтим маску
        
    if event == cv2.EVENT_RBUTTONDOWN: #Правой кнопкой мышки стираем
        drawing = 2
        XY_mask = cv2.circle(XY_mask, point, radius,0,-1)
        mask[current_img] = XY_mask
                  
        
    if event == cv2.EVENT_MBUTTONDOWN: #Средней кнопкой мышки делаем заливку
        #drawing = 3
        m = np.zeros((IMG_WIDTH+2, IMG_WIDTH+2), np.uint8)
        cv2.floodFill(XY_mask, None, point, drawing_color)
        mask[current_img] = XY_mask
                  
    elif event == cv2.EVENT_MOUSEMOVE: #Если будем двигать мышки с зажатой кнопкой, оно будет рисовать/стирать
        if drawing == 1:
            XY_mask = cv2.circle(XY_mask, point, radius,drawing_color,-1)
            mask[current_img] = XY_mask
        if drawing == 2:
            XY_mask = cv2.circle(XY_mask, point, radius,0,-1)
            mask[current_img] = XY_mask
        cursor_img[:,:] = 0
        cursor_img = cv2.circle(cursor_img, point, radius,140,-1)
            
    elif event == cv2.EVENT_LBUTTONUP: #При отпускании кнопки перестаём рисовать
        drawing = 0
        
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = 0
    
    
    
    
    Show_XY_image(current_img)
    
    
def Pass(val):
    pass
        
def Save_images():
    for i in tqdm(range(len(mask))):
        XY_mask  = mask[i]
        XY_mask = cv2.normalize(XY_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)   
        cv2.imwrite(path + '/Mask_vessel/'  + onlyfiles[i], XY_mask)
    
def Mask_alpha_slider(val):
    global Mask_alpha
    Mask_alpha = val/10
    Show_XY_image(current_img)
    print(Mask_alpha)

path = 'C:/Users/Taran/Desktop/PREFUL_1200/olyafirst1200/Png'
#path = 'C:/Users/Taran/Desktop/OP_NS_1800/Png'
onlyfiles = listdir(path + '/Image')
onlyfiles.sort(key=len)
print(onlyfiles)
#images = np.empty(len(onlyfiles), dtype=object)
images = []
mask = []
for n in range(0, len(onlyfiles)):
  #images[n] = cv2.imread( join(mypath,onlyfiles[n]) , 2)
  images.append(cv2.imread( join(path + '/Image',onlyfiles[n]) , 2))
IMG_WIDTH = len(images[0])


if not os.path.exists(path + '/Mask_vessel/'):
    os.mkdir(path + '/Mask_vessel/')
    mask = np.zeros((len(images), IMG_WIDTH, IMG_WIDTH), np.uint8)
    mask = np.asarray(mask)
    for i in tqdm(range(len(mask))):
        XY_mask  = mask[i]
        XY_mask = cv2.normalize(XY_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(path + '/Mask_vessel/'  + onlyfiles[i], XY_mask)

mask = []
for n in range(0, len(onlyfiles)):
  mask.append(cv2.imread( join(path + '/Mask_vessel',onlyfiles[n]) , 2))
images = np.asarray(images)
mask = np.asarray(mask)


current_img = 0

drawing = 0

Mask_alpha = 0.5


cursor_img = np.zeros((IMG_WIDTH,IMG_WIDTH), np.uint8)




cv2.namedWindow("TrackBar2")
cv2.resizeWindow("TrackBar2", 500, 80)
cv2.createTrackbar('Radius', 'TrackBar2', 0, 11, Pass)
cv2.createTrackbar('Mask_alpha', 'TrackBar2', 5, 10, Mask_alpha_slider)
#cv2.createButton("myButtonName",stack_play_pause,cv2.CV_PUSH_BUTTON,1);
#cv2.setWindowProperty("TrackBar", cv2.WND_PROP_TOPMOST, 1)






Show_XY_image(0)
cv2.setMouseCallback("XY+mask", Mouse_draw)
view = 0

while(1):
    full_key_code = cv2.waitKeyEx(0)
    if full_key_code == 119:
        current_img += 1
        if current_img >= len(images):
            current_img = len(images)-1
        Show_XY_image(current_img)
        print('image ' + str(current_img) + ' / ' + str(len(images)))
    if full_key_code == 113:
        current_img -= 1        
        if current_img < 0:
            current_img = 0
        Show_XY_image(current_img)
        print('image ' + str(current_img) + ' / ' + str(len(images)))
    if full_key_code == 32:
        Save_images()
        print('images saved!')
    if full_key_code == 118: #V    Вкл/выкл отображение маски
        if view == 0:
            XY_img  = images[current_img]
            XY_img = cv2.normalize(XY_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
            cv2.imshow("XY+mask", XY_img)
            view = 1   
        elif  view == 1:    
            Show_XY_image(current_img)
            view = 0
    if full_key_code == 99: #C
        mask[current_img] =  mask[current_img-1]
        Show_XY_image(current_img)