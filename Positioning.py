import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import glob
import random
print random.random()
import shutil
import pandas as pd
import win32api, win32con, win32gui

im_orig = cv2.imread("screen_capture.jpg")

cv2.startWindowThread()
cv2.imshow("Keypoints", im_orig)
cv2.waitKey()
cv2.destroyAllWindows()
height, width, channels = im_orig.shape
print height, width, channels

# define functions that extract sub images
import argparse
refPt = []
cropping = False
mouseX = -1; mouseY = -1
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest

def draw_circle(event,x,y,flags,param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(image, (x, y), 100, (255, 0, 0), -1)
        mouseX,mouseY = x, y

def new_rect(mouseX, mouseY, df):
    full_df = df[df.area == "full"]
    df1 = df[df.area != "full"]
    df1['gapx'] = mouseX - df1.x;  #df1[df1['gapx']<0].gapx = 1000 #penalize negatives
    df1['gapx1'] = df1.x1 - mouseX;# df1[df1['gapx1']<0].gapx1 = 1000
    df1['gapy'] = mouseY - df1.y; #df1[df1['gapy']<0].gapy = 1000
    df1['gapy1'] = df1.y1 - mouseY; #df1[df1['gapy1']<0].gapy1 = 1000
    df1['gap_min'] = df1[['gapx', 'gapx1', 'gapy','gapy1']].min(axis = 1)
    ix = df1['gap_min'].argmax()
    if df1.gap_min[ix] > 0:
        if df1.gapx[ix] == df1.gap_min[ix]:
            df1.x[ix] = mouseX
            df1.x1[ix] = mouseX + df1.w[ix]
        if df1.gapx1[ix] == df1.gap_min[ix]:
            df1.x1[ix] = mouseX
            df1.x[ix] = mouseX - df1.w[ix]
        if df1.gapy[ix] == df1.gap_min[ix]:
            df1.y[ix] = mouseY
            df1.y1[ix] = mouseY + df1.h[ix]
        if df1.gapy1[ix] == df1.gap_min[ix]:
            df1.y1[ix] = mouseY
            df1.y[ix] = mouseY - df1.h[ix]
    df1.drop(['gapx', 'gapx1', 'gapy', 'gapy1', 'gap_min'], inplace = True, axis = 1)
    df1 = pd.concat([full_df, df1])
    return df1

def move_x(mouseX, mouseY, df):
    full_df = df[df.area == "full"]
    df1 = df[df.area != "full"]
    ix = (df1.y > mouseY) | (df1.y1 < mouseY)

    df1['gap1'] = abs(mouseX - df1.x); #df1[df1['gapy']<0].gapy = 1000
    df1['gap2'] = abs(df1.x1 - mouseX); #df1[df1['gapy1']<0].gapy1 = 1000
    df1.gap1[ix] = 100000; df1.gap2[ix] = 100000
    df1['gap_min'] = df1[['gap1', 'gap2']].min(axis = 1)
    print df1.gap_min[ix]


    ix = df1['gap_min'].argmin()
    print "min gap"
    print ix
    if df1.gap_min[ix] > 0:
        if df1.gap1[ix] == df1.gap_min[ix]:
            df1.x[ix] = mouseX
        if df1.gap2[ix] == df1.gap_min[ix]:
            df1.x1[ix] = mouseX
    df1.drop(['gap1','gap2', 'gap_min'], inplace = True, axis = 1)
    df1 = pd.concat([full_df, df1])
    return df1

def move_y(mouseX, mouseY,df):
    full_df = df[df.area == "full"]
    df1 = df[df.area != "full"]
    ix = (df1.x > mouseX) | (df1.x1 < mouseX)

    df1['gap1'] = abs(mouseY - df1.y);
    df1['gap2'] = abs(df1.y1 - mouseY);
    df1.gap1[ix] = 100000; df1.gap2[ix] = 100000
    df1['gap_min'] = df1[['gap1', 'gap2']].min(axis = 1)

    ix = df1['gap_min'].argmin()
    print ix
    if df1.gap_min[ix] > 0:
        if df1.gap1[ix] == df1.gap_min[ix]:
            df1.y[ix] = mouseY
        if df1.gap2[ix] == df1.gap_min[ix]:
            df1.y1[ix] = mouseY
    df1.drop(['gap1', 'gap2', 'gap_min'], inplace = True, axis = 1)
    df1 = pd.concat([full_df, df1])
    return df1

df = pd.read_csv('templates/bovada_abs_position_v3.csv')
# rescale image based on window size and template positions
min_x = 0; min_y =0
max_x = width; max_y = height
delta_x = min_x - min(df.x)
delta_y = min_y - min(df.y)
print delta_x, delta_y, min_x, min_y, min(df.x), min(df.y)
df['x'] = df['x'].apply(lambda x: x+delta_x)
df['x1'] = df['x1'].apply(lambda x: x+delta_x)
df['y'] = df['y'].apply(lambda x: x+delta_y)
df['y1'] = df['y1'].apply(lambda x: x+delta_y)

cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_and_crop)
cv2.setMouseCallback('image', draw_circle)
# keep looping until the 'q' key is pressed
image = im_orig.copy()
while True:
    image = im_orig.copy()
    # display the image and wait for a keypress
    for ii in range(len(df)):
        this_row = df.iloc[[ii]]
        cv2.putText(image, this_row.area.iloc[0], (this_row.x, this_row.y), 1, 1, (0, 255, 0), 1)
        cv2.rectangle(image, (this_row.x, this_row.y), (this_row.x1, this_row.y1), (255, 0, 0), 2)
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = im_orig.copy()
    if key == ord("p"):
        # print mouseX,mouseY
        df = new_rect(mouseX, mouseY, df)
    if key == ord("x"):
        df = move_x(mouseX, mouseY, df)
    if key == ord("y"):
        df = move_y(mouseX, mouseY, df)
    # if the 'c' key is pressed, break from the loop
    elif key == ord("b"):
        break
   
# close all open windows
cv2.destroyAllWindows()
