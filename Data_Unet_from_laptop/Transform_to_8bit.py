#C:/Users/Taran/AppData/Local/Programs/Python/Python39/python.exe -i "$(FULL_CURRENT_PATH)"
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import cv2
from tqdm import tqdm
import sys
import os

import argparse
from pathlib import Path
#path = 'C:/Users/Taran/Desktop/Data_Unet/View_test/Mask'
#path = 'C:/Users/Taran/Desktop/PREFUL_1200/olyafirst1200/Image'

#path = 'C:/Users/Taran/Desktop/BLEOM/BLEOMICINE/13rat/ExpRaw_13'
#path = 'C:/Users/Taran/Desktop/mct/12/12_4/data_insp'
#parent_path = os.path.dirname(path)
#folder_name = os.path.relpath(path, parent_path)
#folder_name = 'Mask_insp'
#folder_name = 'Mask_exp'
#folder_name = 'Image_insp'
#folder_name = 'Image_exp'

PATH = []
FOLDER_NAME = []

#PATH.append('C:/Users/Taran/Desktop/BLEOM/CONTROL/21rat_bad/ExpRaw')
#PATH.append('C:/Users/Taran/Desktop/BLEOM/CONTROL/21rat_bad/InspRaw')
#PATH.append('C:/Users/Taran/Desktop/BLEOM/CONTROL/21rat_bad/EXP')
#PATH.append('C:/Users/Taran/Desktop/BLEOM/CONTROL/21rat_bad/INSP')

#FOLDER_NAME.append('Image_exp')
#FOLDER_NAME.append('Image_insp')
#FOLDER_NAME.append('Mask_exp')
#FOLDER_NAME.append('Mask_insp')

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="display a square of a given number")
#parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
args = parser.parse_args()
if args.path:
    PATH.append(os.path.normpath(args.path))
    FOLDER_NAME.append(os.path.basename(args.path))
    print(PATH)
    print(FOLDER_NAME)
else:
    print('path = none')

"""
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/18rat/EXP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/18rat/INSP_DATA')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/18rat/MASK_EXP')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/18rat/MASK_INSP')
PATH.append('K:\Data_Unet/Rat/Olya_26_07_23/TREATMENT/18rat/MASK_PAT')


FOLDER_NAME.append('EXP_DATA')
FOLDER_NAME.append('INSP_DATA')
FOLDER_NAME.append('MASK_EXP')
FOLDER_NAME.append('MASK_INSP')
FOLDER_NAME.append('MASK_PAT')
"""

def Convert_to_8bit(path, folder_name):


    if not os.path.exists(path):
        raise ValueError('Path not found!')
    
    parent_path = os.path.dirname(path)
    #print(parent_path)
    onlyfiles = listdir(path)
    onlyfiles = [f for f in onlyfiles if os.path.isfile(path+'/'+f)] 
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
      images[n] = cv2.imread( join(path,onlyfiles[n]) , 2)

    #k = 0
    print('Converting images...')
    if not os.path.exists(parent_path + '/Png/'):
        os.mkdir(parent_path + '/Png/')
    if not os.path.exists(parent_path + '/Png/' + folder_name):
        os.mkdir(parent_path + '/Png/' + folder_name)
    for n in tqdm(range(len(images))):
        img_8bit = cv2.normalize(images[n], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(parent_path + '/Png/' + folder_name + '/img_' + str(n) + '.png', img_8bit)
    print('Done!')





for i in range(len(PATH)):
    print('\nConverting folder ' + str(i+1) + '/' + str(len(PATH)) + '...')
    Convert_to_8bit(PATH[i], FOLDER_NAME[i])
print('\nConversion completed!')


