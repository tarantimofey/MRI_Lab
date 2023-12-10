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



def L_adaptive_a_slider(val):
    #image = sorted_images[0]
    L_img = img.copy()
    cv2.rectangle(L_img,(len(img)//2,0),(len(img),len(img)),0,-1)
    image = L_img
    if val%2 == 0:
        val += 1
        cv2.setTrackbarPos('L_adaptive a', 'TrackBar', val)
    b = cv2.getTrackbarPos('L_adaptive b', 'TrackBar')
    th = mp.Make_mask_adaptive(image, val, b)
    cv2.imshow('th', th)
    global L_th1
    L_th1 = Refine_mask(th)
    #cv2.imshow('th1', L_th1)
    Show_images()
    
def L_adaptive_b_slider(val):
    #image = sorted_images[0]
    L_img = img.copy()
    cv2.rectangle(L_img,(len(img)//2,0),(len(img),len(img)),0,-1)
    image = L_img
    a = cv2.getTrackbarPos('L_adaptive a', 'TrackBar')
    th = mp.Make_mask_adaptive(image, a, val)
    cv2.imshow('th', th)
    global L_th1
    L_th1 = Refine_mask(th)
    #cv2.imshow('th1', L_th1)
    Show_images()
    
def L_threshold_slider(val):
    #image = sorted_images[0]
    L_img = img.copy()
    cv2.rectangle(L_img,(len(img)//2,0),(len(img),len(img)),0,-1)
    image = L_img
    th = mp.Make_mask(image, val)
    cv2.imshow('th', th)
    global L_th1
    L_th1 = Refine_mask(th)
    #cv2.imshow('th1', L_th1)
    Show_images()
    
def R_adaptive_a_slider(val):
    #image = sorted_images[0]  
    R_img = img.copy()
    cv2.rectangle(R_img,(0,0),(len(img)//2,len(img)),0,-1)  
    image = R_img
    if val%2 == 0:
        val += 1
        cv2.setTrackbarPos('R_adaptive a', 'TrackBar', val)
    b = cv2.getTrackbarPos('R_adaptive b', 'TrackBar')
    th = mp.Make_mask_adaptive(image, val, b)
    cv2.imshow('th', th)
    global R_th1
    R_th1 = Refine_mask(th)
    #cv2.imshow('th1', R_th1)
    Show_images()
    
def R_adaptive_b_slider(val):
    #image = sorted_images[0]  
    R_img = img.copy()
    cv2.rectangle(R_img,(0,0),(len(img)//2,len(img)),0,-1)  
    image = R_img
    a = cv2.getTrackbarPos('R_adaptive a', 'TrackBar')
    th = mp.Make_mask_adaptive(image, a, val)
    cv2.imshow('th', th)
    global R_th1
    R_th1 = Refine_mask(th)
    #cv2.imshow('th1', R_th1)
    Show_images()
    
def R_threshold_slider(val):
    R_img = img.copy()
    cv2.rectangle(R_img,(0,0),(len(img)//2,len(img)),0,-1)  
    image = R_img
    th = mp.Make_mask(image, val)
    cv2.imshow('th', th)
    global R_th1
    R_th1 = Refine_mask(th)
    #th1 = L_th1 + R_th1
    #th1 = R_th1
    #cv2.imshow('th1', th1)
    Show_images()

def Refine_mask(th):
    th_c = th.copy()
    for i in range(0, len(th_c)):
        clear=255
        cv2.floodFill(th_c, None, (i, 0), 255)
        cv2.floodFill(th_c, None, (i, len(th_c)-1), 255)
        cv2.floodFill(th_c, None, (0, i), 255)
        cv2.floodFill(th_c, None, (len(th_c)-1, i), 255)
    th_c = cv2.normalize(th_c, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #print(th_c)
    th_c = cv2.bitwise_not(th_c)
    cnt, hierarchy = cv2.findContours(th_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(th_c1, cnt, -1, 100, 1)
    hull_list = []
    j=0
    for i in range(0, len(cnt)):
        if(cv2.contourArea(cnt[i]) < 500):
            #print(cv2.contourArea(cnt[i]))
            hull = cv2.convexHull(cnt[i])        
            hull_list.append(hull)
            #cv2.fillPoly(imageread,hull_list[i],100);
            cv2.drawContours(th_c, hull_list, j, 255, -1)
            j+=1
            
            
    th_c = cv2.bitwise_not(th_c)
    cnt, hierarchy = cv2.findContours(th_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(th_c1, cnt, -1, 100, 1)
    hull_list = []
    j=0
    for i in range(0, len(cnt)):
        if(cv2.contourArea(cnt[i]) < 500):
            #print(cv2.contourArea(cnt[i]))
            hull = cv2.convexHull(cnt[i])        
            hull_list.append(hull)
            #cv2.fillPoly(imageread,hull_list[i],100);
            cv2.drawContours(th_c, hull_list, j, 255, -1)
            j+=1
    #th_c = cv2.dilate(th_c,(3,3),iterations = 1)   
    med = cv2.getTrackbarPos('Median', 'TrackBar') 
    if med%2 == 0:
        med += 1
    th_c = cv2.medianBlur(th_c, med)
    
    th_c = cv2.bitwise_not(th_c)
    return th_c

def Show_images():
    global th1
    th1 = L_th1 + R_th1
    cv2.imshow('th1', th1)
    th1_color = th1.copy()
    th1_color = cv2.cvtColor(th1_color, cv2.COLOR_GRAY2RGB)
    
    for i in range(0, len(th1_color)):
        for j in range(0, len(th1_color)):
            th1_color[i][j][0] = 0
            th1_color[i][j][1] = 0       
    
    img_8bit = img.copy()
    img_8bit = cv2.normalize(img_8bit, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
    superimposed = cv2.addWeighted(img_8bit, 1, th1_color, 0.5, 0)
    
    cv2.namedWindow('Img+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Img+mask", superimposed)
    cv2.imshow("Img", img_8bit)

def Update_image():
    mode = cv2.getTrackbarPos('Mode', 'TrackBar')
    if mode == 0:
        L_adaptive_a_slider(cv2.getTrackbarPos('L_adaptive a', 'TrackBar'))
        L_adaptive_b_slider(cv2.getTrackbarPos('L_adaptive b', 'TrackBar'))
        R_adaptive_a_slider(cv2.getTrackbarPos('R_adaptive a', 'TrackBar'))
        R_adaptive_b_slider(cv2.getTrackbarPos('R_adaptive b', 'TrackBar'))
    if mode == 1:
        L_threshold_slider(cv2.getTrackbarPos('L_threshold', 'TrackBar'))
        R_threshold_slider(cv2.getTrackbarPos('R_threshold', 'TrackBar'))
    Show_images()

def Save_images(path, n):
    #Tiff_path = path + '/Images'
    #Mask_path = path + '/Masks'
    Image = img
    Mask = th1
    #cv2.imwrite('C:/Users/Taran/Desktop/img.png', img)
    #cv2.imwrite(path + '/Mask/2d_seq_' + str(n) + '.png', th1)
    cv2.imwrite(path + '/Mask/' + onlyfiles[n], th1)
    """
    for i in range(0, len(images)):
        
        cv2.imwrite('C:/Users/Taran/Desktop/img/VR_3.png', VR_3)
    """
    
def Mode_slider(val):
    Update_image()
    
def Median_slider(val):
    Update_image()

def Mouse_draw(event, x, y, flags, param):
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        global th1
        th1 = cv2.rectangle(th1,(x,y),(x+5,y+5),255,-1)
        cv2.imshow('th1', th1)
    """
    global ix, iy, drawing, img
    global th1
    #drawing = 0
    point = (x, y)
    radius = cv2.getTrackbarPos('Radius', 'TrackBar2')
      
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = 1
        ix = x
        iy = y            
        #th1 = cv2.rectangle(th1, point,point,255,-1)
        th1 = cv2.circle(th1, point, radius,255,-1)
        
    if event == cv2.EVENT_RBUTTONDOWN:
        drawing = 2
        ix = x
        iy = y            
        #th1 = cv2.rectangle(th1, point,point,0,-1)
        th1 = cv2.circle(th1, point, radius,0,-1)
              
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == 1:
            #th1 = cv2.rectangle(th1, point,point,255,-1)
            th1 = cv2.circle(th1, point, radius,255,-1)
        if drawing == 2:
            #th1 = cv2.rectangle(th1, point,point,0,-1)
            th1 = cv2.circle(th1, point, radius,0,-1)
        cv2.imshow('th1', th1)
      
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = 0
        
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = 0
    cv2.imshow('th1', th1)
    th1_color = th1.copy()
    th1_color = cv2.cvtColor(th1_color, cv2.COLOR_GRAY2RGB)
    
    for i in range(0, len(th1_color)):
        for j in range(0, len(th1_color)):
            th1_color[i][j][0] = 0
            th1_color[i][j][1] = 0       
    
    img_8bit = img.copy()
    img_8bit = cv2.normalize(img_8bit, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
    superimposed = cv2.addWeighted(img_8bit, 1, th1_color, 0.5, 0)
    
    #cv2.namedWindow('Img+mask', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Img+mask", superimposed)
    #cv2.imshow("Img", img_8bit)
        
def Pass(val):
    pass
        

#seq = cv2.imread('C:/Users/Taran/Desktop/2dseq.tif', 2)

path = 'C:/Users/Taran/Desktop/Ready_data/OP_SS/Png'
mypath = path + '/Image'
onlyfiles = listdir(mypath)
onlyfiles.sort(key=len)
print(onlyfiles)
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) , 2)


current_img_stack = 0
img = images[current_img_stack]
img_8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.namedWindow('Img', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Img", img_8)
#cv2.resizeWindow('Img', 200, 200)
cv2.namedWindow('th', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('th1', cv2.WINDOW_KEEPRATIO)

#L_th1 = img
#R_th1 = img











cv2.namedWindow("TrackBar")
cv2.resizeWindow("TrackBar", 640, 350)
threshold_max = 2**8
#cv2.createTrackbar('slider0', 'TrackBar', 0, len(sorted_array)-1, on_change)
#cv2.createTrackbar('slider1', 'TrackBar', 0, len(sorted_array)-1, stack_control)
cv2.createTrackbar('L_adaptive a', 'TrackBar', 70, 600, L_adaptive_a_slider)
cv2.createTrackbar('L_adaptive b', 'TrackBar', 9, 100, L_adaptive_b_slider)
cv2.createTrackbar('L_threshold', 'TrackBar', 0, threshold_max, L_threshold_slider)
cv2.createTrackbar('R_adaptive a', 'TrackBar', 41, 600, R_adaptive_a_slider)
cv2.createTrackbar('R_adaptive b', 'TrackBar', 8, 100, R_adaptive_b_slider)
cv2.createTrackbar('R_threshold', 'TrackBar', 0, threshold_max, R_threshold_slider)
cv2.createTrackbar('Mode', 'TrackBar', 0, 1, Mode_slider)
cv2.createTrackbar('Median', 'TrackBar', 0, 11, Median_slider)

cv2.namedWindow("TrackBar2")
cv2.resizeWindow("TrackBar2", 500, 50)
cv2.createTrackbar('Radius', 'TrackBar2', 0, 11, Pass)
#cv2.createButton("myButtonName",stack_play_pause,cv2.CV_PUSH_BUTTON,1);
#cv2.setWindowProperty("TrackBar", cv2.WND_PROP_TOPMOST, 1)


"""
cv2.waitKey(0)
cv2.destroyAllWindows()
"""




cv2.setMouseCallback("th1", Mouse_draw)

while(1):
    full_key_code = cv2.waitKeyEx(0)
    if full_key_code == 2555904:
        current_img_stack += 1
        if current_img_stack >= len(images):
            current_img_stack = len(images)-1
        img = images[current_img_stack]
        Update_image()
        print('image ' + str(current_img_stack))
    if full_key_code == 2424832:
        current_img_stack -= 1        
        if current_img_stack < 0:
            current_img_stack = 0
        img = images[current_img_stack]
        Update_image()
        print('image ' + str(current_img_stack))
    if full_key_code == 32:
        Save_images(path, current_img_stack)
        print('image ' + str(current_img_stack) + ' saved!')