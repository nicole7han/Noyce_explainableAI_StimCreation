import os, cv2, json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import random
from scipy.stats import norm, multivariate_normal
from script.utils import *

''' This script define different rules and sample feature prototype values'''

'''
There are four objects: ellipse, circle, triangle, recetnagle
Step 1. choose primary object and randomly choose a mean prototype from 27 posibilities (3 values for each feature)
parameters = {
    # ellipse
    'e_width1': [[30,40,50], 10], #[[means], difference between two categories]
    'e_length2': [[70,80,90], 10],
    'e_orient3': [[0,45,90], 10],

    # circle
    'c_radius1': [[40,50,60], 10],
    'c_contrast2': [[.4,.5,.6], .1],
    'c_border3': [[5,7,9], 2],

    # triangle
    't_height1': [[70,80,90], 10],
    't_length2': [[30,40,50], 10],
    't_orient3': [[0,45,90], 10],

    # rectangle
    'r_length1': [[70,80,90], 10],
    'r_contrast2': [[.4,.5,.6], .1],
    'r_orient3': [[0,45,90], 10],

    # sd for gaussian distribution
    'sd':5,

    # mean distance between objects
    'dis': [300,400,500]
    
Step 2. choose secondary object from the remaining 3 objects
pick one feature out of three to determine Catergory A vs. Category B, the distributions of the other 2 features are the same for Categroy A and Catgeory B
e.g., pick triangle height, then randomly choose value for category A: 70
choose a mean for category B: 70-10 or 70+10

Step 3. choose relative distance object from the remaining 2 objects
use distance to determin Catgeory A vs. Catergoy B. the distributions of the 3rd features are the same for Categroy A and Catgeory B
e.g., randomly pick distance for Categroy A: [40, 60, 80], 
choose a mean for category B: e.g., 40-10 or 40+10

Step 4. choose the remaining object as the irrelevant object, which is not informative about the category
randomly sample one mean prototype out of 27 possibilities
same distribution for categroy A and category B

'''

imageSize = 225*3
parameters = {
    # ellipse
    'e_width1': [[20,30,40], 10], #[means, difference between two categories]
    'e_length2': [[60,65,70], 10],
    'e_oriented3': [[0,45,90], 10],
    'ecov':[[10,0,0],[0,10,0],[0,0,10]], #gaussian covariance matrix for three features

    # circle
    'c_radius1': [[50,60,70], 10],
    'c_contrast2': [[.5,.6,.7], .1],
    'c_width3': [[5,7,9], 2],
    'ccov':[[10,0,0],[0,.05,0],[0,0,1]],

    # triangle
    't_height1': [[90,100,110], 10],
    't_base2': [[35,40,55], 10],
    't_oriented3': [[0,45,90], 10],
    'tcov':[[10,0,0],[0,10,0],[0,0,10]],

    # rectangle
    'r_length1': [[80,90,100], 10],
    'r_width2': [[20,30,50], 10],
    'r_oriented3': [[0,45,90], 10],
    'rcov':[[10,0,0],[0,10,0],[0,0,10]],

    # mean distance
    'dis': [150,200,250],
    'dis_sd': 20,
}


''' STEP 1: Choose primary object, all 3 features matter'''
N = 200 #number of different rules
rule = pd.DataFrame()
for i in range(N):
    row_info = {}
    row_info['rule'] = 'rule{}'.format(i+1)

    objects = ['e','c','t','r']
    idx = random.choice(range(len(objects)))
    obj1 = objects[idx]
    objects.pop(idx) # update object list

    # randomly choose prototype mean for Category A
    obj1_A = [random.choice(val[0]) for (k,val) in parameters.items() if obj1+'_' in k]
    obj1_diff = [val[1] for (k,val) in parameters.items() if obj1+'_' in k]
    obj1_B = []
    for x,d in zip(obj1_A, obj1_diff):
        if random.random()>0.5: # randomly choose prototype mean for Category B
            obj1_B.append(x+d)
        else:
            obj1_B.append(x-d)
    row_info['obj1'] = obj1
    row_info['{}_f1_A'.format(obj1)],row_info['{}_f2_A'.format(obj1)],row_info['{}_f3_A'.format(obj1)] = obj1_A
    row_info['{}_f1_B'.format(obj1)],row_info['{}_f2_B'.format(obj1)],row_info['{}_f3_B'.format(obj1)] = obj1_B

    ''' STEP 2: CHoose secondary object, only 1 feature matter'''
    idx = random.choice(range(len(objects)))
    obj2 = objects[idx]
    objects.pop(idx) # update object list
    # randomly choose feature prototype mean for Category A
    obj2_A = [random.choice(val[0]) for (k,val) in parameters.items() if obj2+'_' in k]
    obj2_diff = [val[1] for (k, val) in parameters.items() if obj2 + '_' in k]
    obj2_B = obj2_A.copy()
    # randomly choose on feature that is used to discriminate Categroy A vs. B
    f_idx = random.choice(range(3))
    if random.random()>0.5: # randomly choose prototype mean for Category B
        obj2_B[f_idx] = obj2_A[f_idx] + obj2_diff[f_idx]
    else:
        obj2_B[f_idx] = obj2_A[f_idx] - obj2_diff[f_idx]
    row_info['obj2'] = obj2
    row_info['obj2_feat'] = [k for (k,v) in parameters.items() if obj2+'_' in k and str(f_idx+1) in k][0]
    row_info['{}_f1_A'.format(obj2)],row_info['{}_f2_A'.format(obj2)],row_info['{}_f3_A'.format(obj2)] = obj2_A
    row_info['{}_f1_B'.format(obj2)], row_info['{}_f2_B'.format(obj2)], row_info['{}_f3_B'.format(obj2)] = obj2_B

    ''' STEP 3: Choose relative distance object, only distance matters'''
    idx = random.choice(range(len(objects)))
    obj3 = objects[idx]
    objects.pop(idx) # update object list
    # randomly choose feature prototype mean for Category A
    obj3_A = [random.choice(val[0]) for (k,val) in parameters.items() if obj3+'_' in k]
    dis_A = random.choice(parameters['dis'])
    if random.random()>0.5: # randomly choose prototype mean for Category B
        dis_B = dis_A + 50
    else:
        dis_B = dis_A - 50
    row_info['obj3'] = obj3
    row_info['{}_f1_A'.format(obj3)], row_info['{}_f2_A'.format(obj3)], row_info['{}_f3_A'.format(obj3)] = obj3_A
    row_info['{}_f1_B'.format(obj3)], row_info['{}_f2_B'.format(obj3)], row_info['{}_f3_B'.format(obj3)] = obj3_A
    row_info['obj3_dis_A'] = dis_A
    row_info['obj3_dis_B'] = dis_B

    ''' STEP 4: Choose irrelevant object'''
    obj4 = objects.pop()
    # randomly choose feature prototype mean for Category A
    obj4_A = [random.choice(val[0]) for (k,val) in parameters.items() if obj4+'_' in k]
    row_info['obj4'] = obj4
    row_info['{}_f1_A'.format(obj4)], row_info['{}_f2_A'.format(obj4)], row_info['{}_f3_A'.format(obj4)] = obj4_A
    row_info['{}_f1_B'.format(obj4)], row_info['{}_f2_B'.format(obj4)], row_info['{}_f3_B'.format(obj4)] = obj4_A
    rule = rule.append(row_info, ignore_index=True)

out_path = 'Stim_MultipleRules'
os.makedirs(out_path, exist_ok=True)
with open('{}/parameters.json'.format(out_path), 'w') as fp:
    json.dump(parameters, fp)
rule.to_excel('{}/rule_parameters.xlsx'.format(out_path), index=None)