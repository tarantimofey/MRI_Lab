import os
from os import listdir
from os.path import join
import cv2
import numpy as np
  
 
 
 
 
import argparse
from pathlib import Path



path_main = ''

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="enter path")
#parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
args = parser.parse_args()
if args.path:
    path_main = os.path.normpath(args.path)
else:
    print('path = none')
    exit()
 

if not os.path.exists(path_main):
    raise ValueError('Path not found!')

#path_main = 'C:/Users/Taran/Desktop/Maximovskaya_Anastasia/LOC'



    
image_ids = []
temp = listdir(path_main)
temp.sort(key=len)
#image_ids.append(temp)
image_ids = temp

images = []
for i in range(len(image_ids)):
    img = cv2.imread(join(path_main,image_ids[i]) , 2)
    images.append(img)

images = np.asarray(images)

bright_images_indeces = []
for i in range(len(images)):
    #if images[i].mean()/images.min() > 3:
    if images[i].mean() > 1000:
        print(i, images[i].mean(), '<-----')
        bright_images_indeces.append(i)
    else:
        print(i, images[i].mean())

print('Bright images found:', len(bright_images_indeces))

response = input('Do you wish to convert and save images?(Y/n) ')
if response == 'n':
    print('Ok.')
    exit()
if response != 'n' and response != 'y':
    print('No viable responce recieved. Exiting the programm')
    exit()


for i in bright_images_indeces:
    images[i] = images[i]/10


path_Tiff_norm = path_main + '/Tiff_norm'
if not os.path.exists(path_Tiff_norm):
    os.makedirs(path_Tiff_norm)


for i in range(len(images)):
    cv2.imwrite(path_Tiff_norm + '/' + 'Img_' + str(i) + '.tif', images[i])

print('Done!')
