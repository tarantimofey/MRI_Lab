#C:/Users/Taran/AppData/Local/Programs/Python/Python39/python.exe -i "$(FULL_CURRENT_PATH)"

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
import math 
import My_Preful_lib.My_Preful_lib as mp
from os import listdir
from os.path import isfile, join
import cv2
import imageio
from brukerapi.dataset import Dataset

import math
import plotly.graph_objects as go

import os
import random
from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import open3d as o3d

import win32api
import win32con


IMG_WIDTH = 156 #Размер входных данных
IMG_HEIGHT = 156
IMG_DEPTH = 156
IMG_CHANNELS = 1

WINDOW_SIZE = 853 #Размер окон с проекциями

paths = []
paths.append('C:/Users/Taran/Desktop/rat_5/LUNGS.h64/2/pdata/1/') #0 Добавляем в массив пути к данным для разных крысок

paths.append('C:/Users/Taran/Desktop/rat_5/LUNGS.ha4/3/pdata/1/') #1

paths.append('C:/Users/Taran/Desktop/rat_5/LUNGS.hh3/2/pdata/1/') #2
paths.append('C:/Users/Taran/Desktop/rat_5/LUNGS.hh3/3/pdata/1/') #3

paths.append('C:/Users/Taran/Desktop/rat_5/LUNGS.ho3/3/pdata/1/') #4
paths.append('C:/Users/Taran/Desktop/rat_5/LUNGS.ho3/4/pdata/1/') #5
path = paths[5] #Выбираем крыску

rotation = 0 #Оставляем 0. Если 1, то будет вращать картинки




def Show_XY_image(n): #Функция для апдейта картинки в окне
    global XY_img 
    XY_img  = d[:, :, n] #Берём слайс из трёхмерного массива данных
    XY_img = cv2.normalize(XY_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    #cv2.namedWindow('XY', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow("XY", XY_img)
    
    XY_img_color = XY_img.copy()
    XY_img_color = cv2.cvtColor(XY_img_color, cv2.COLOR_GRAY2RGB)
    
    XY_mask = mask[:, :, n] #Берём слайс из трёхмерного массива маски
    #XY_mask = cv2.normalize(XY_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    XY_mask_color = XY_mask.copy()
    XY_mask_color = cv2.cvtColor(XY_mask_color, cv2.COLOR_GRAY2RGB)
    XY_mask_color[XY_mask_color[:,:,0] == 1] = (0,0,255)
    XY_mask_color[XY_mask_color[:,:,0] == 2] = (0,255,0)
    XY_mask_color[XY_mask_color[:,:,0] == 3] = (255,0,0)
    superimposed = cv2.addWeighted(XY_img_color, 1, XY_mask_color, 0.5, 0) #Накладываем изоражение маски на изображение крыски
    cv2.namedWindow('XY+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("XY+mask", superimposed)   #Апдейтим картинку
    cv2.resizeWindow("XY+mask", WINDOW_SIZE, WINDOW_SIZE)
    print(XY_mask)
        
def Show_XZ_image(n):
    current_img_XZ = n
    XZ_img = d[:, n, :]
    XZ_img = cv2.normalize(XZ_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    #cv2.namedWindow('XZ', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow("XZ", XZ_img)
    
    XZ_img_color = XZ_img.copy()
    XZ_img_color = cv2.cvtColor(XZ_img_color, cv2.COLOR_GRAY2RGB)
    
    XZ_mask = mask[:, n, :]
    #XZ_mask = cv2.normalize(XZ_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    XZ_mask_color = XZ_mask.copy()
    XZ_mask_color = cv2.cvtColor(XZ_mask_color, cv2.COLOR_GRAY2RGB)
    XZ_mask_color[XZ_mask_color[:,:,0] == 1] = (0,0,255)
    XZ_mask_color[XZ_mask_color[:,:,0] == 2] = (0,255,0)
    XZ_mask_color[XZ_mask_color[:,:,0] == 3] = (255,0,0)
    superimposed = cv2.addWeighted(XZ_img_color, 1, XZ_mask_color, 0.5, 0)
    cross_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
    cross_img = cv2.cvtColor(cross_img, cv2.COLOR_GRAY2RGB)
    cross_img = cv2.rectangle(cross_img,(current_img_XY,0),(current_img_XY,IMG_HEIGHT),(0,255,0),-1)
    cross_img = cv2.rectangle(cross_img,(0, current_img_YZ),(IMG_WIDTH, current_img_YZ),(255,0,0),-1)
    superimposed = cv2.addWeighted(superimposed, 1, cross_img, 0.2, 0)
    cv2.namedWindow('XZ+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("XZ+mask", superimposed)  
    cv2.resizeWindow("XZ+mask", WINDOW_SIZE, WINDOW_SIZE)

def Show_YZ_image(n):
    current_img_YZ = n
    YZ_img = d[n, :, :]
    YZ_img = cv2.normalize(YZ_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    #cv2.namedWindow('YZ', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow("YZ", YZ_img)
    
    YZ_img_color = YZ_img.copy()
    YZ_img_color = cv2.cvtColor(YZ_img_color, cv2.COLOR_GRAY2RGB)
    
    YZ_mask = mask[n, :, :]
    #YZ_mask = cv2.normalize(YZ_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    
    YZ_mask_color = YZ_mask.copy()
    YZ_mask_color = cv2.cvtColor(YZ_mask_color, cv2.COLOR_GRAY2RGB)
    YZ_mask_color[YZ_mask_color[:,:,0] == 1] = (0,0,255)
    YZ_mask_color[YZ_mask_color[:,:,0] == 2] = (0,255,0)
    YZ_mask_color[YZ_mask_color[:,:,0] == 3] = (255,0,0)
    superimposed = cv2.addWeighted(YZ_img_color, 1, YZ_mask_color, 0.5, 0)
    cross_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
    cross_img = cv2.cvtColor(cross_img, cv2.COLOR_GRAY2RGB)
    cross_img = cv2.rectangle(cross_img,(current_img_XY,0),(current_img_XY,IMG_HEIGHT),(0,255,0),-1)
    cross_img = cv2.rectangle(cross_img,(0, current_img_XZ),(IMG_WIDTH, current_img_XZ),(255,0,0),-1)
    superimposed = cv2.addWeighted(superimposed, 1, cross_img, 0.2, 0)
    cv2.namedWindow('YZ+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("YZ+mask", superimposed)     
    cv2.resizeWindow("YZ+mask", WINDOW_SIZE, WINDOW_SIZE)

drawing = 0
def Mouse_draw_XY(event, x, y, flags, param): #Функция для рисовашек
    #win32api.SetCursor(None)
    win32api.SetCursor(win32api.LoadCursor(0, win32con.OCR_NORMAL))
    global ix, iy, drawing
    #drawing = 0
    point = (x, y) #Поучаем координаты курсора
    radius = cv2.getTrackbarPos('Radius', 'TrackBar2') #Получаем значение радиуса из бегунка
    #radius = 2
    n = current_img_XY    
    cursor_img = np.zeros((IMG_HEIGHT,IMG_WIDTH), np.uint8)
    
    XY_mask = np.zeros((IMG_HEIGHT,IMG_WIDTH), np.uint8)
    #XY_mask = mask[:, :, n] #Получаем слайс маски
    #XY_mask = mask[:, :, n].copy() #Получаем слайс маски
    XY_mask = cv2.addWeighted(XY_mask, 1, mask[:, :, n], 1, 0) 
    #XY_mask = cv2.normalize(XY_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  
    #XY_mask = XY_mask.copy()
    #XY_mask = np.asarray(XY_mask)
    #print(drawing_color)
    
    if event == cv2.EVENT_LBUTTONDOWN: #Левой кнопкой мышки рисуем
        drawing = 1
        XY_mask = cv2.circle(XY_mask, point, radius,drawing_color,-1)
        mask[:, :, n] = XY_mask #Апдейтим маску
        
    if event == cv2.EVENT_RBUTTONDOWN: #Правой кнопкой мышки стираем
        drawing = 2
        XY_mask = cv2.circle(XY_mask, point, radius,0,-1)
        mask[:, :, n] = XY_mask
                  
        
    if event == cv2.EVENT_MBUTTONDOWN: #Средней кнопкой мышки делаем заливку
        #drawing = 3
        m = np.zeros((IMG_HEIGHT+2, IMG_WIDTH+2), np.uint8)
        cv2.floodFill(XY_mask, None, point, drawing_color)
        mask[:, :, n] = XY_mask
                  
    elif event == cv2.EVENT_MOUSEMOVE: #Если будем двигать мышки с зажатой кнопкой, оно будет рисовать/стирать
        if drawing == 1:
            XY_mask = cv2.circle(XY_mask, point, radius,drawing_color,-1)
            mask[:, :, n] = XY_mask
        if drawing == 2:
            XY_mask = cv2.circle(XY_mask, point, radius,0,-1)
            mask[:, :, n] = XY_mask
        cursor_img = cv2.circle(cursor_img, point, radius,140,-1)
            
    elif event == cv2.EVENT_LBUTTONUP: #При отпускании кнопки перестаём рисовать
        drawing = 0
        Show_XZ_image(current_img_XZ) #Заодно апдейтим картинки в других проекциях
        Show_YZ_image(current_img_YZ)
        
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = 0
      
    XY_img  = d[:, :, n] #Апдейтим картинку
    Show_XY_image(current_img_XY)
    """
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
    cursor_img = cv2.cvtColor(cursor_img, cv2.COLOR_GRAY2RGB)
    superimposed = cv2.addWeighted(superimposed, 1, cursor_img, 0.5, 0)
    cv2.namedWindow('XY+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("XY+mask", superimposed)  
    """


def Mouse_draw_XZ(event, x, y, flags, param):
    global ix, iy, drawing
    #drawing = 0
    point = (x, y)
    radius = cv2.getTrackbarPos('Radius', 'TrackBar2')
    #radius = 2
    n = current_img_XZ
    cursor_img = np.zeros((IMG_HEIGHT,IMG_WIDTH), np.uint8)
    XZ_mask = np.zeros((IMG_HEIGHT,IMG_WIDTH), np.uint8)    
    #XZ_mask = mask[:, n, :]
    #XZ_mask = cv2.normalize(XZ_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)      
    XZ_mask = cv2.addWeighted(XZ_mask, 1, mask[:, n, :], 1, 0) 
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = 1
        XZ_mask = cv2.circle(XZ_mask, point, radius,drawing_color,-1)
        mask[:, n, :] = XZ_mask
        
    if event == cv2.EVENT_RBUTTONDOWN:
        drawing = 2
        XZ_mask = cv2.circle(XZ_mask, point, radius,0,-1)
        mask[:, n, :] = XZ_mask
                  
        
    if event == cv2.EVENT_MBUTTONDOWN:
        #drawing = 3
        m = np.zeros((IMG_HEIGHT+2, IMG_WIDTH+2), np.uint8)
        cv2.floodFill(XZ_mask, None, point, drawing_color)
        mask[:, n, :] = XZ_mask
                  
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == 1:
            XZ_mask = cv2.circle(XZ_mask, point, radius,drawing_color,-1)
            mask[:, n, :] = XZ_mask
        if drawing == 2:
            XZ_mask = cv2.circle(XZ_mask, point, radius,0,-1)
            mask[:, n, :] = XZ_mask
        cursor_img = cv2.circle(cursor_img, point, radius,140,-1)
        #cursor_img = cv2.cvtColor(cursor_img, cv2.COLOR_GRAY2RGB)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = 0
        Show_XY_image(current_img_XY)
        Show_YZ_image(current_img_YZ)
        
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = 0
      
    XZ_img  = d[:, n, :]
    Show_XZ_image(current_img_XZ)


def Mouse_draw_YZ(event, x, y, flags, param):
    global ix, iy, drawing
    #drawing = 0
    point = (x, y)
    radius = cv2.getTrackbarPos('Radius', 'TrackBar2')
    #radius = 2
    n = current_img_YZ
    cursor_img = np.zeros((IMG_HEIGHT,IMG_WIDTH), np.uint8)
    YZ_mask = np.zeros((IMG_HEIGHT,IMG_WIDTH), np.uint8)    
    #YZ_mask = mask[n, :, :]
    #YZ_mask = cv2.normalize(YZ_mask, None, 0, drawing_color, cv2.NORM_MINMAX, cv2.CV_8U)       
    YZ_mask = cv2.addWeighted(YZ_mask, 1, mask[n, :, :], 1, 0) 
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = 1
        YZ_mask = cv2.circle(YZ_mask, point, radius,drawing_color,-1)
        mask[n, :, :] = YZ_mask
        
    if event == cv2.EVENT_RBUTTONDOWN:
        drawing = 2
        YZ_mask = cv2.circle(YZ_mask, point, radius,0,-1)
        mask[n, :, :] = YZ_mask
                  
        
    if event == cv2.EVENT_MBUTTONDOWN:
        #drawing = 3
        m = np.zeros((IMG_HEIGHT+2, IMG_WIDTH+2), np.uint8)
        cv2.floodFill(YZ_mask, None, point, drawing_color)
        mask[n, :, :] = YZ_mask
                  
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == 1:
            YZ_mask = cv2.circle(YZ_mask, point, radius,drawing_color,-1)
            mask[n, :, :] = YZ_mask
        if drawing == 2:
            YZ_mask = cv2.circle(YZ_mask, point, radius,0,-1)
            mask[n, :, :] = YZ_mask
        cursor_img = cv2.circle(cursor_img, point, radius,140,-1)
        #cursor_img = cv2.cvtColor(cursor_img, cv2.COLOR_GRAY2RGB)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = 0
        Show_XY_image(current_img_XY)
        Show_YZ_image(current_img_XZ)
        
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = 0
      
    YZ_img  = d[n, :, :]
    Show_YZ_image(current_img_YZ)

def Pass(val):
    pass
        
def Save_images(p): #Функция для сохранения серии XY изображений
    path = p
    for i in tqdm(range(len(d))):
        XY_img  = d[:, :, i]
        XY_img = cv2.normalize(XY_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)   
        cv2.imwrite(path + '/png/XY/Image/XY_img_' + str(i) + '.png', XY_img)
        XY_mask  = mask[:, :, i]
        XY_mask = cv2.normalize(XY_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)   
        cv2.imwrite(path + '/png/XY/Mask/XY_img_' + str(i) + '.png', XY_mask)


dataset = Dataset(path+'2dseq')    # create data set, works for fid, 2dseq, rawdata.x, sers
#X = dataset.data                         # access data array
#dataset.VisuCoreSize                 # get a value of a single parameter
d = np.asarray(dataset.data)
d = dataset.data[:, :, :, 0] #Создаём трёхмерный массив данных
#print(d.shape)

#global mask
mask = np.zeros((IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), np.uint8)
"""
for i in range(0, 156):
    for j in range(0, 156):
        for k in range(0, 156):
            if math.sqrt((i-78)**2+(j-78)**2+(k-78)**2) < 20:
                mask[i,j,k] = 1
for i in range(78, 98):
    for j in range(78, 98):
        for k in range(78, 98):
            mask[i,j,k] = 1
"""


X, Y, Z = np.mgrid[0:156, 0:156, 0:156]






#Swap(d, 1, 2)
#d[0,1,2] = d[1,0,2]
#d = np.swapaxes(d,1,2) 
#d = np.swapaxes(d,0,1)
#d = d.T
if rotation == 1:
    d = np.rot90(d, 1, (0,1))
    d = np.flip(d, 0)


img_XY = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
img_XZ = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
img_YZ = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)









def Chsnge_drawing_color(x):
    drawing_color = x

cv2.namedWindow("TrackBar2")
cv2.resizeWindow("TrackBar2", 500, 200)
cv2.createTrackbar('Radius', 'TrackBar2', 0, 11, Pass)
#cv2.createTrackbar('XY', 'TrackBar2', 0, IMG_DEPTH, Show_XY_image)
cv2.createTrackbar('XZ', 'TrackBar2', 0, IMG_HEIGHT, Show_XZ_image)
cv2.createTrackbar('YZ', 'TrackBar2', 0, IMG_WIDTH, Show_YZ_image)
cv2.createTrackbar('drawing_color', 'TrackBar2', 0, 255, Chsnge_drawing_color)





current_img = 0
current_img_XY = 0
current_img_XZ = 0
current_img_YZ = 0
img_len = len(X)
Show_XY_image(current_img_XY)
Show_XZ_image(current_img_XZ)
Show_YZ_image(current_img_YZ)


cv2.setMouseCallback("XY+mask", Mouse_draw_XY)
cv2.setMouseCallback("XZ+mask", Mouse_draw_XZ)
cv2.setMouseCallback("YZ+mask", Mouse_draw_YZ)

view = 0
drawing_color = 255

while(1):
    full_key_code = cv2.waitKeyEx(0)
    if full_key_code == 119: #2555904 W/right    Следующая XY картинка
        current_img_XY += 1
        if current_img_XY >= img_len:
            current_img_XY = img_len-1          
        Show_XY_image(current_img_XY)
        print('image XY ' + str(current_img_XY) + ' / ' + str(img_len))
    if full_key_code == 113: #2424832 Q/left    Предыдущая XY картинка
        current_img_XY -= 1        
        if current_img_XY < 0:
            current_img_XY = 0
        Show_XY_image(current_img_XY)
        print('image XY ' + str(current_img_XY) + ' / ' + str(img_len))
    if full_key_code == 115: #S    Следующая XZ картинка
        current_img_XZ += 1
        if current_img_XZ >= img_len:
            current_img_XZ = img_len-1          
        Show_XZ_image(current_img_XZ)
        print('image XZ ' + str(current_img_XZ) + ' / ' + str(img_len))
    if full_key_code == 97: #A    Предыдущая XZ картинка
        current_img_XZ -= 1        
        if current_img_XZ < 0:
            current_img_XZ = 0
        Show_XZ_image(current_img_XZ)
        print('image XZ ' + str(current_img_XY) + ' / ' + str(img_len))
    if full_key_code == 120: #X    Следующая YZ картинка
        current_img_YZ += 1
        if current_img_YZ >= img_len:
            current_img_YZ = img_len-1          
        Show_YZ_image(current_img_YZ)
        print('image YZ ' + str(current_img_YZ) + ' / ' + str(img_len))
    if full_key_code == 122: #Z    Предыдущая YZ картинка
        current_img_YZ -= 1        
        if current_img_YZ < 0:
            current_img_YZ = 0
        Show_YZ_image(current_img_YZ)
        print('image YZ ' + str(current_img_YZ) + ' / ' + str(img_len))
    if full_key_code == 109: #M    Строит 3д модельку
        fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=mask.flatten(),
        isomin=0.1,
        isomax=10,
        opacity=1,
        surface_count=20,
        ))
        fig.show()
    if full_key_code == 118: #V    Вкл/выкл отображение маски
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
    if full_key_code == 32: #Space    Сохраняем маску в файл
        """
        cv2.imwrite('C:/Users/Taran/Desktop/' + 'Img' + str(n) + '.png', superimposed)
        print('image ' + str(current_img) + ' saved!')
        """
        #cv2.imwrite('C:/Users/Taran/Desktop/' + 'VR_Data.png', VR_Data)
        #print('VR_Data.png saved!')
        with open(path+'mask.npy', 'wb') as f:
            with np.printoptions(threshold=np.inf):
                #f.write(str(mask))
                np.save(f, mask)
        print('mask.npy saved!')
    #plt.show(block=False)
    if full_key_code == 114: #R    Загружаем маску из файла
        with open(path+'mask.npy', 'rb') as f:
            with np.printoptions(threshold=np.inf):
                #mask = f.read()
                mask = np.load(f)                
                if rotation == 1:
                    mask = np.rot90(mask, 1, (0,1))
                    mask = np.flip(mask, 0) 
        print('mask.npy loaded')
    if full_key_code == 105: #I    Сохраняем серию XY картинок (Но для этого нужно вручную создать путь (см. функцию Save_images)). Потом допишу, чтобы он создавался автоматически
        Save_images(path)
        print('XY images saved!')
        
    if full_key_code == 49: #1
        drawing_color = 1
    if full_key_code == 50: #1
        drawing_color = 2
    if full_key_code == 51: #1
        drawing_color = 3