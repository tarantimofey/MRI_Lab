import cv2
import numpy as np


mask_lung = cv2.imread('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/Png/MASK_EXP/img_86.png', 2)
mask_pat = cv2.imread('/media/taran/SSD2/Data_Unet/Rat/Olya_26_07_23/BLEOMICINE/13rat/Png/MASK_PAT/img_86.png', 2)

img_combined = np.zeros((152,152), np.uint8)
img_combined[mask_lung==255] = 1
img_combined[mask_pat==255] = 2



cv2.namedWindow('img_combined', cv2.WINDOW_KEEPRATIO)
cv2.imshow("img_combined", img_combined)
cv2.waitKey()