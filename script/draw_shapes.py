import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import random
from script.utils import *

def add_noise(stim, sigma=.06):
    stim[stim == 1] = 140 / 255  # value 1 -> 140/255
    stim[stim == 0] = .5  # background 0 -> 0.5
    stimuli = stim + sigma * np.random.randn(stim.shape[0], stim.shape[1])
    return stimuli


def draw_shape(f1, f2, f3, shape, loc=[0.5,0.5], imageSize=255*3, image=None):
    '''
    Create single stimuli
    :param f1, f2, f3: feature values of the stimuli
    :type f1: int/float
    :param shape: type of the shape "e": ellipse, 't':triangle, 'c': circle, 'r':rectangular
    :type f1: str
    :param loc: spatial location of the stimuli [x,y], ranging from [0,1]
    :type loc: numpy array
    :param imageSize: imageSize  height, width
    :type imageSize: int
    :param image: numpy image
    :type image: numpy array
    '''
    locx, locy = int(loc[0] * imageSize), int(loc[1] * imageSize)
    check_overlap = False
    overlap = 0
    try: # draw with existing objects
        image_orig = image.copy()
        check_overlap = True
    except:
        pass
    image = np.zeros([imageSize, imageSize])

    if shape == 'e': #ellipse
        # f1: wdth, f2: length, f3:orientation
        x = np.linspace(0, imageSize-1, imageSize).astype(int)
        y = np.linspace(0, imageSize-1, imageSize).astype(int)
        columnsInImage, rowsInImage = np.meshgrid(x, y)
        b = f2 #length
        a = f1 #width
        theta = (f3)*np.pi/180
        image = ( ( (columnsInImage - locx)*np.cos(theta)+(rowsInImage - locy)*np.sin(theta) )**2/a**2 + \
                ( ( columnsInImage - locx)*np.sin(theta)-(rowsInImage - locy)*np.cos(theta) )**2/b**2 <= 1).astype(float)
    elif shape == 'c': #circle
        # f1: radius, f2: contrast, f3:border thickness\
        cir_r, cir_con, border = round(f1), f2, round(f3)
        cir_l = 128*(1+cir_con)/255 #circle luminance based on contrast
        image = cv2.circle(image, (locx, locy), cir_r, [cir_l, cir_l, cir_l], border)
    elif shape == 't': #triangle
        # f1: height, f2: base, f3:orientation
        trian_h, trian_b, trian_orien = f1, f2, f3
        top = rotate([locx, round(locy-trian_h/2)], origin=[locx,locy], degrees=round(trian_orien))
        bottomleft, bottomright = rotate([round(locx-trian_b/2), round(locy+trian_h/2)],origin=[locx,locy],degrees=round(trian_orien)),\
                                  rotate([round(locx+trian_b/2), round(locy+trian_h/2)],origin=[locx,locy],degrees=round(trian_orien))
        points = [np.array(top), np.array(bottomleft), np.array(bottomright)]
        points = np.array(points).astype('int64')
        image = cv2.fillPoly(image, [points] , True, 1)

    elif shape == 'r': # rectangle
        # f1: length, f2: width, f3:orientation
        rec_length, rec_width, rec_orien = f1, f2, f3
        rec_topleft, rec_bottomright = rotate([round(locx-rec_width/2), round(locy-rec_length/2)],origin=[locx,locy],degrees=round(rec_orien)),\
                                  rotate([round(locx+rec_width/2), round(locy+rec_length/2)],origin=[locx,locy],degrees=round(rec_orien))
        rec_topright, rec_bottomleft = rotate([round(locx + rec_width / 2), round(locy - rec_length / 2)],
                                              origin=[locx, locy], degrees=round(rec_orien)), \
                                       rotate([round(locx - rec_width / 2), round(locy + rec_length / 2)],
                                              origin=[locx, locy], degrees=round(rec_orien))
        points = [np.array(rec_topleft), np.array(rec_bottomleft),np.array(rec_bottomright), np.array(rec_topright)]
        points = np.array(points).astype('int64')
        image = cv2.fillPoly(image, [points] , True, 1)

    if check_overlap:
        overlap = (image*image_orig).sum()
        if overlap==0:
            image = image+image_orig
        else:
            overlap=1

    try:
        return image, image_orig, overlap
    except:
        return image