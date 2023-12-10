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
from sklearn.model_selection import train_test_split
import os
import random
from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize

import os

PATH = 'C:/Users/Taran/Desktop/Data_Unet/View_test/Mask'
"""
if __name__ == "__main__":
    for (root,dirs,files) in os.walk('C:/Users/Taran/Desktop/Data_Unet/View_test', topdown=True):
        print (root)
        print (dirs)
        print (files)
        print ('--------------------------------')
"""        
        
#img_ids = [ f for f in listdir(PATH + '/Image') if isfile(join(PATH + '/Image',f)) ]
img_ids = listdir(PATH)
img_ids.sort(key=len) 
print(img_ids)