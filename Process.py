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

height, width, channels = im_orig.shape
print height, width, channels

df = pd.read_csv('templates/bovada_abs_position_v3.csv')
# rescale image based on window size and template positions
#print df[1:2]
#print rect
min_x = 0; min_y =0
max_x = width; max_y = height
delta_x = min_x - min(df.x)
delta_y = min_y - min(df.y)
print delta_x, delta_y, min_x, min_y, min(df.x), min(df.y)
df['x'] = df['x'].apply(lambda x: x + delta_x)
df['x1'] = df['x1'].apply(lambda x: x + delta_x)
df['y'] = df['y'].apply(lambda x: x + delta_y)
df['y1'] = df['y1'].apply(lambda x: x + delta_y)

# last step is for adjusting difference and geting needed parameters
# the following starts real capturing data
if os.path.exists('./real_time/areas'):
    shutil.rmtree('./real_time/areas')
os.makedirs('./real_time/areas')

# channel 2 just to get back of cards
im_orig = cv2.imread("screen_capture.jpg")

columns = ['x', 'y', 'x1', 'y1', 'classification', 'score', 'area']

res_df = pd.DataFrame(columns = columns)
print res_df
# print df
for ii in range(len(df)):
    this_row = df.iloc[[ii]]
    columns = ['x', 'y', 'x1', 'y1', 'classification', 'score', 'area']
    area_df = pd.DataFrame(columns = columns)
    # print this_row.area.to_string(index=False) == 'full'
    area = this_row.area.to_string(index=False)
    this_area_w = this_row.x1 - this_row.x
    this_area_h = this_row.y1 - this_row.y
    if area == "full":
        continue
    # print this_row.area
    roi=im_orig[this_row.y:this_row.y1, this_row.x:this_row.x1]
    this_dir = './real_time/areas/'+area

    if (os.path.exists(this_dir) == False):
        os.makedirs(this_dir)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if area == 'p1stack':
        roi_gray = 255 - roi_gray
        cv2.startWindowThread()
        cv2.imshow("Keypoints", roi_gray)
        cv2.waitKey()
        cv2.destroyAllWindows()
        print roi_gray

    path = this_dir + '/' + area + '.jpg'
    threshold = 180
    if "ard" in area:
        threshold = 140
    #if "bet" in area or "Pot" in area:
    #    threshold = 180
    flag, roi_th = cv2.threshold(roi_gray, threshold, 255, cv2.THRESH_BINARY)

    cv2.imwrite(path, roi_th)

    image, contours, hierarchy = cv2.findContours(roi_th.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse = True)
    im_rect = roi_th.copy()
    idx = 0
    for ct in contours:
        idx = idx + 1
        x,y,w,h = cv2.boundingRect(ct)
        #print w, h
        #print this_area_w, this_area_h, w/this_area_w
        if w > 2 or h > 3:
            sub_roi = im_rect[y : y+h, x : x+w]
            p_str = str(x)+"_"+str(y)+"_"+str(w)+"_"+str(h)
            path = this_dir + '/' + str(p_str) + '.jpg'
            cv2.imwrite(path, sub_roi)

            height = 10; width = 10
            im_th = cv2.resize(sub_roi,(width, height), interpolation = cv2.INTER_CUBIC)
            this_max = 0
            if "status" in area:
                subdir = "status"
            elif "mainPot" == area:
                subdir = "mainPot"
            elif "ard" in area:
                subdir = "card"
            elif "bet" in area:
                subdir = "bet"
            elif "status" in area:
                subdir = "status"
            elif "stack" in area:
                subdir = "stack"
            elif "deal" in area:
                subdir = "dealer"
            elif "totalPot" in area:
                subdir = "totalPot"
            elif "control" in area:
                subdir = "control"

            templatesdir = "./templates/final/" + subdir
            for itemp in glob.glob(templatesdir + "/*.jpg"):
                template = cv2.imread(itemp)
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                flag, template = cv2.threshold(template, 120, 255, cv2.THRESH_BINARY)
                template = cv2.resize(template,(width, height), interpolation = cv2.INTER_CUBIC)
                #print template.shape

                res1 = cv2.matchTemplate(template, im_th, cv2.TM_CCOEFF_NORMED )
                res2 = cv2.matchTemplate(im_th, template, cv2.TM_CCOEFF_NORMED )
                if(np.max(res1) > np.max(res2)):
                    res = res2
                else:
                    res = res1
                # print res, np.max(res)
                if(this_max < np.max(res)):
                    this_max = np.max(res)
                    max_temp = itemp
                    best_temp = template.copy()
            #print "best tmp", best_temp
            #print "target_image", im_th
            #print this_max
            if this_max > 0.85:
                classification = max_temp.replace(".jpg", "").replace(templatesdir, "").replace("\\", "")
                classification = classification.split('_', 1)[0]
                classification = classification.replace("card", "").replace("digit", "").replace("letter", "")

                new_row = len(area_df); area_df.loc[new_row,0 : 4] =  np.array([x, y, x+w, y+h])
                area_df.loc[new_row].classification = classification; area_df.loc[new_row].score = this_max
                area_df.loc[new_row].area = area
    if len(area_df) == 0:
        continue

    area_df = area_df.sort(['x', 'y'], ascending=[True, True])
    if "ard" in area:
        area_df["h"] = area_df.y1 - area_df.y
        area_df = area_df[area_df.h >= 0.7 * area_df["h"].max()]
        area_df["xy"] = 2 * area_df.x + area_df.y
        area_df = area_df.sort(['xy'], ascending = [True])
    if "control" in area:
        area_df["h"] = area_df.y1 - area_df.y
        area_df = area_df[area_df.h >= 0.5 * area_df["h"].max()]
        area_df["xy"] = 2 * area_df.x + area_df.y
        area_df = area_df.sort(['xy'], ascending = [True])
        # print area_df
    if area == "totalPot": # so smaller shape is not in the objects
         print area_df
    str_class = ""
    for jj in range(len(area_df)):
        this_area_df_row = area_df.iloc[[jj]]
        if jj >= 1:
            if ((this_area_df_row.x >= area_df.iloc[[jj - 1]].x) &
                (this_area_df_row.x1 <= area_df.iloc[[jj - 1]].x1) &
                (this_area_df_row.y >= area_df.iloc[[jj - 1]].y) &
                (this_area_df_row.y1 <= area_df.iloc[[jj - 1]].y1)).bool():
                 continue
        str_class = str_class + this_area_df_row.classification.iloc[0]
        #print str_class
    new_row = len(res_df); res_df.loc[new_row, 0:4] = np.array([x, y, x + w, y + h])
    res_df.loc[new_row].classification = str_class;
    res_df.loc[new_row].score = 0
    res_df.loc[new_row].area = area

print "finished"
        # adding rectangles to original image
        # cv2.rectangle(im_rect, (x, y), (x + w, y + h), (255, 0, 0), 1)
print res_df