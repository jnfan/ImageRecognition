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

#### Find the coordinates of the window of interest using string match #####

def isRealWindow(hWnd):
    #Return True iff given window is a real Windows application window.
    if not win32gui.IsWindowVisible(hWnd):
        return False
    if win32gui.GetParent(hWnd) != 0:
        return False
    hasNoOwner = win32gui.GetWindow(hWnd, win32con.GW_OWNER) == 0
    lExStyle = win32gui.GetWindowLong(hWnd, win32con.GWL_EXSTYLE)
    if (((lExStyle & win32con.WS_EX_TOOLWINDOW) == 0 and hasNoOwner)
      or ((lExStyle & win32con.WS_EX_APPWINDOW != 0) and not hasNoOwner)):
        if win32gui.GetWindowText(hWnd):
            return True
    return False
rect = []
def callback(hwnd, extra):
    global rect
    if not isRealWindow(hwnd):
        return
    title = win32gui.GetWindowText(hwnd)
    # print title
    if "CV" in title:
        rect = win32gui.GetWindowRect(hwnd)
        x = rect[0]
        y = rect[1]
        w = rect[2] - x
        h = rect[3] - y
        print "Window %s:" % win32gui.GetWindowText(hwnd)
        print "\tLocation: (%d, %d)" % (x, y)
        print "\t    Size: (%d, %d)" % (w, h)

win32gui.EnumWindows(callback, None)
print rect

## extract the image from the coordinates obtained from last session
from PIL import ImageGrab

im_orig = ImageGrab.grab(rect)
im_orig.show()
im_orig.save("screen_capture.jpg", 'JPEG', subsampling=0, quality=100)
# ImageGrab.grab(rect).save("screen_capture.jpg", "JPEG")