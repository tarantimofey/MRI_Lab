import pydicom 
import pydicom.data 
import os
from os import listdir
import cv2
  
 
 
 
 
import argparse
from pathlib import Path



path_main = ''

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="enter path")
#parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
args = parser.parse_args()
if args.path:
    path_main = os.path.normpath(args.path)
    print(path_main)
else:
    print('path = none')
  
 

#path_main = 'C:/Users/Taran/Desktop/Maximovskaya_Anastasia/LOC'


path_DICOM = path_main + '/DICOM'
path_Tiff = path_main + '/Tiff'
path_Png = path_main + '/Png'

if not os.path.exists(path_Tiff):
    os.makedirs(path_Tiff)
if not os.path.exists(path_Png):
    os.makedirs(path_Png)


image_ids = []
temp = listdir(path_DICOM)
temp.sort(key=len)
#image_ids.append(temp)
image_ids = temp



for i in range(len(image_ids)):
    print(image_ids[i])
    ds = pydicom.dcmread(path_DICOM + '/' + image_ids[i])
    cv2.imwrite(path_Tiff + '/' + 'Img_' + str(i) + '.tif', ds.pixel_array)
    img_png = cv2.normalize(ds.pixel_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(path_Png + '/' + 'Img_' + str(i) + '.png', img_png)
    
print('Done!')

#ds = pydicom.dcmread(path_DICOM + '/Грудная_клетка_GE_FFE_3_20231205_1348100.dcm') 

