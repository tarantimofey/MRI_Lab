import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
import math 
import cv2
"""
import My_Preful_lib.My_Preful_lib as mp
from os import listdir
from os.path import isfile, join
import imageio
from brukerapi.dataset import Dataset

import math
import plotly.graph_objects as go

import os
import sys
import random
from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import open3d as o3d


import win32api
"""

IMG_WIDTH = 156
IMG_HEIGHT = 156
IMG_DEPTH = 156
IMG_CHANNELS = 1
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", help="display a square of a given number")
#parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
args = parser.parse_args()
if args.path:
    path = args.path
else:
    print('path = none')

from pathlib import Path
data_folder = Path(path)
#file_to_open = data_folder / "raw_data.txt"

print(data_folder)
import subprocess
subprocess.Popen(r'explorer /select,%s' %data_folder)
"""
"""
print(os.get_terminal_size())


perf_signal =  [9.964848, 13.394535, 6.737247, 6.2413898, 12.1250725, 13.18221, 6.093972, 13.585094, 15.375343, 9.433812, 10.855095, 8.227775, 10.899943, 12.803188, 11.180971, 13.020653, 9.779725, 12.263847, 6.4790974, 13.605124, 14.993285, 7.8995795, 13.081422, 13.796658, 13.730177, 6.9966364, 12.999466, 14.936197, 9.026756, 10.968069, 9.075634, 10.877248, 13.168833, 7.9620075, 12.084021, 9.938988, 12.709679, 6.459284, 13.149271, 6.158165, 13.194704, 15.562082, 8.649532, 11.385743, 14.467241, 8.470192, 13.8568945, 11.428749, 7.70697, 12.397342, 6.0531673, 14.8345785, 7.0618124, 11.415402, 14.753886, 8.518544, 7.700244, 12.929716, 7.2430773, 11.424782, 6.4035597, 13.056719, 13.543034, 6.2989497, 12.0385, 13.684262, 6.8473735, 11.309364, 13.068511, 9.857428, 11.382526, 13.465616, 6.6098948, 11.641247, 13.330527, 6.647041, 13.086932, 13.522629, 13.478211, 9.560099, 11.494731, 14.329451, 7.5436106, 11.728827, 14.002108, 10.339604, 13.487302, 7.6842175, 12.468956, 8.99345, 14.428659, 7.902732, 12.141168, 14.779984, 8.831863, 11.569307, 7.3229814, 12.347151, 13.147777, 7.4157834, 11.347826, 11.113546, 12.263979, 6.66465, 13.946079, 13.264342, 9.858188, 12.914696, 6.7778025, 14.43964, 16.27258, 6.1230383, 6.8697653, 12.623537, 14.909271, 7.894567, 12.111072, 13.880574, 8.021898, 12.067232, 6.644071, 13.305101, 11.209416, 15.151649, 7.028459, 12.204426, 14.993687, 8.994333, 12.164, 7.229787, 12.57141, 14.033065, 8.185642, 13.3068905, 12.207918, 14.340294, 9.609922, 12.830992, 7.2760115, 12.0049515, 11.972876, 9.086252, 11.4187155, 12.917207, 7.174416, 10.966381, 11.951605, 6.9698553, 13.4641, 15.185959, 7.608886, 12.275789, 7.4707036, 11.619531, 14.853485, 11.596175, 7.808454, 11.313039, 15.056139, 9.657525, 11.555294, 7.0071836, 13.826544, 14.898036, 8.728516, 11.693349, 6.8242636, 14.340103, 10.304018, 11.985485, 7.45183, 13.035833, 13.519569, 9.8920765, 11.777086, 13.32882, 8.886697, 13.104084, 12.596761, 8.706079, 11.985714, 15.175917, 7.501152, 12.350946, 15.258948, 8.133366, 13.249851, 8.844268, 14.673088, 11.757296, 15.049569, 7.150384, 12.32808, 14.004805, 7.732542, 13.414148, 9.667935, 11.082182, 14.172104]

perf_signal = np.asarray(perf_signal)
perf_signal = perf_signal-perf_signal.mean()

def line_filter_func(n, c):
    c_n = int(c*n/2)
    y = np.ones(n, dtype=np.float)
    for i in range(0, c_n):
        y[i]=i/c_n
    for i in range(0, c_n):
        y[n-1-i]=i/c_n
    #print(y)
    return y


def fft_hipass(y, c, o):
    y_fft = np.fft.fft(y)
    #n_to_delete = int(c*len(y_fft)/2)
    for i in range(o):
        y_fft = y_fft*line_filter_func(len(y_fft), c)
    
    y_ifft = np.fft.ifft(y_fft)
    fig_fft = plt.figure(figsize=(14, 2), num='Perfusion signal FFT')
    ax_fft = plt.subplot(1,1,1, label='FFT')
    ax_fft.plot(y_fft.real, color='tab:blue')
    ax_fft.plot(y_fft.imag, color='tab:orange')

    return y_ifft

from matplotlib.widgets import Button, Slider


# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line_0, = ax.plot(perf_signal, 'o-', linewidth=0.5, label='data', markersize=2, color='tab:blue', alpha=0.5)
line, = ax.plot(fft_hipass(perf_signal, 0.5, 1), 'o-', linewidth=0.5, label='data', markersize=2, color='tab:orange', alpha=0.5)
ax.set_xlabel('Time [s]')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
cutoff_slider = Slider(
    ax=axfreq,
    label='Cutoff frequency [Hz]',
    valmin=0,
    valmax=1,
    valinit=0.5,
)
axorder = fig.add_axes([0.25, 0.05, 0.65, 0.03])
order_slider = Slider(
    ax=axorder,
    label='Order',
    valmin=1,
    valmax=6,
    valinit=2,
    valstep=1,
)

# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(fft_hipass(perf_signal, cutoff_slider.val, order_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
cutoff_slider.on_changed(update)
order_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    cutoff_slider.reset()
button.on_clicked(reset)

plt.show()


#print()
"""
"""
import win32con
def mouse_evt(event, x, y, flags, param):
# could also probably do: def mouse_evt(*args):
    win32api.SetCursor(None)
win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_SIZEALL))
    
"""

cv2.namedWindow("Img")


img = np.zeros((500,500), np.uint8)
img[200:300, 250:300] = 255
cv2.imshow('Img',img)
#cv2.setMouseCallback("Img", Mouse_draw_XY)
#cv2.setMouseCallback("Img", mouse_evt)




"""
arr =  img.flatten()
for i in img:
    for j in i:
        j = 255
    #print(i)


def Freeq_gen(C, R1, R2):
    x = []
    y = []
    for i in range(1, 100):
        x.append(R2/i)
        y.append(1/(2*np.pi*C*(R1+R2/i)))
    plt.plot(x, y)
    plt.show(block=False)
    
    y1 = 1/(2*np.pi*C*R1)
    y2 = 1/(2*np.pi*C*(R1+R2/2))
    y3 = 1/(2*np.pi*C*(R1+R2))
    print(y1, '-------', y2, '-------', y3)
    

C = 4.7*10**(-6)
R1 = 1*10**3
R2 = 500*10**3

Freeq_gen(C, R1, R2)



import pydicom as dicom
import matplotlib.pylab as plt
"""



def func(*elements):
    elements = list(elements)
    for i in range(len(elements)):
        elements[i] = elements[i]*100
    elements = tuple(elements)
    return elements

x1 = 0; x2 = 1; x3 = 2

x1, x2, x3 = func(x1, x2, x3)
print(x1, x2, x3)






while(1):
    k = cv2.waitKey(0)
    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print(k) # else print its value
        
       
