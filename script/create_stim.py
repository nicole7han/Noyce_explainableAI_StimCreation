import os, cv2, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random
from scipy.stats import norm, multivariate_normal
from script.utils import *
from script.draw_shapes import *
os.makedirs('Stim_MultipleRules/Rules_train',exist_ok=True)
os.makedirs('Stim_MultipleRules/Rules_eval',exist_ok=True)

file = open('Stim_MultipleRules/parameters.json')
parameters = json.load(file)
rules = pd.read_excel('Stim_MultipleRules/rule_parameters.xlsx')
sigma = 0.5 # white noise
imageSize = 255*3
max_iter = 10 # maximum iteration to find valid feature values
n_rules = len(rules)
n_eval_rules = 5
for i, param in rules.iterrows():
    print('creating rule {}'.format(i+1))
    n_stim = 20  # 2500
    if i < n_rules-n_eval_rules:  # leave number of rules for evaluation
        n_train = int(n_stim * .8) # 80% training, 20% testing
        outpath = 'Stim_MultipleRules/Rules_train/stim_rule{}'.format(i+1)
        os.makedirs('{}/train'.format(outpath), exist_ok=True)
        os.makedirs('{}/test'.format(outpath), exist_ok=True)
    else:
        n_test = n_stim # all for testing, no training
        n_train = 0
        outpath = 'Stim_MultipleRules/Rules_eval/stim_rule{}'.format(i+1)
        os.makedirs('{}/test'.format(outpath), exist_ok=True)
    os.makedirs(outpath, exist_ok=True)
    stim_info = pd.DataFrame()
    for j in range(n_stim):
        A_info = {}
        A_info['stim'] = 'A{}'.format(j+1) # stim name
        B_info = {}
        B_info['stim'] = 'B{}'.format(j + 1)

        # 1. draw primary object according to the current rule
        # print('creating obj1')
        obj1 = param['obj1']
        A_info['obj1'], B_info['obj1'] = obj1, obj1
        # get category prototype means
        f1Am, f2Am, f3Am = param['{}_f1_A'.format(obj1)], param['{}_f2_A'.format(obj1)], param['{}_f3_A'.format(obj1)]
        f1Bm, f2Bm, f3Bm = param['{}_f1_B'.format(obj1)], param['{}_f2_B'.format(obj1)], param['{}_f3_B'.format(obj1)]
        # sample from gaussian distribution
        f1A, f2A, f3A = np.random.multivariate_normal([f1Am, f2Am, f3Am], parameters['{}cov'.format(obj1)])
        f1B, f2B, f3B = np.random.multivariate_normal([f1Bm, f2Bm, f3Bm], parameters['{}cov'.format(obj1)])
        # primary object default location at the center for the primary object
        imgA = draw_shape(f1A, f2A, f3A, shape=obj1, loc=[0.5, 0.5], imageSize=imageSize)
        imgB = draw_shape(f1B, f2B, f3B, shape=obj1, loc=[0.5, 0.5], imageSize=imageSize)
        A_info['{}_f1'.format(obj1)], A_info['{}_f2'.format(obj1)],A_info['{}_f3'.format(obj1)] = f1A, f2A, f3A
        B_info['{}_f1'.format(obj1)], B_info['{}_f2'.format(obj1)], B_info['{}_f3'.format(obj1)] = f1B, f2B, f3B

        # 2. draw secondary object
        # print('creating obj2')
        obj2 = param['obj2']
        A_info['obj2'], B_info['obj2'] = obj2, obj2
        # get means
        f1Am, f2Am, f3Am = param['{}_f1_A'.format(obj2)], param['{}_f2_A'.format(obj2)], param['{}_f3_A'.format(obj2)]
        f1Bm, f2Bm, f3Bm = param['{}_f1_B'.format(obj2)], param['{}_f2_B'.format(obj2)], param['{}_f3_B'.format(obj2)]
        # sample from gaussian distribution
        f1A, f2A, f3A = np.random.multivariate_normal([f1Am, f2Am, f3Am], parameters['{}cov'.format(obj2)])
        f1B, f2B, f3B = np.random.multivariate_normal([f1Bm, f2Bm, f3Bm], parameters['{}cov'.format(obj2)])

        imgAorig, imgBorig = imgA.copy(), imgB.copy()
        start, overlap, iter = 1, 0, 0
        while start or overlap: # make sure no overlap between objects
            start = 0
            locA = [np.random.uniform(0.15, .85), np.random.uniform(0.15, .85)]
            imgA, imgAorig, overlap = draw_shape(f1A, f2A, f3A, shape=obj2, loc=locA, imageSize=imageSize, image=imgAorig)
            iter+=1
            if iter > max_iter: # resample features
                f1A, f2A, f3A = np.random.multivariate_normal([f1Am, f2Am, f3Am], parameters['{}cov'.format(obj2)])
        start, overlap, iter = 1, 0, 0
        while start or overlap:
            start = 0
            locB = [np.random.uniform(0.15, .85), np.random.uniform(0.15, .85)]
            imgB, imgBorig, overlap = draw_shape(f1B, f2B, f3B, shape=obj2, loc=locB, imageSize=imageSize, image=imgBorig)
            iter += 1
            if iter > max_iter:  # resample features
                f1B, f2B, f3B = np.random.multivariate_normal([f1Bm, f2Bm, f3Bm], parameters['{}cov'.format(obj2)])
        A_info['{}_f1'.format(obj2)], A_info['{}_f2'.format(obj2)],A_info['{}_f3'.format(obj2)] = f1A, f2A, f3A
        B_info['{}_f1'.format(obj2)], B_info['{}_f2'.format(obj2)], B_info['{}_f3'.format(obj2)] = f1B, f2B, f3B
        A_info['obj2_locx'], A_info['obj2_locy'], B_info['obj2_locx'], B_info['obj2_locy'] = list(locA)+list(locB)

        # 3. draw third distance relative object
        # print('creating obj3')
        obj3 = param['obj3']
        A_info['obj3'], B_info['obj3'] = obj3, obj3
        # get means
        f1Am, f2Am, f3Am, disAm = param['{}_f1_A'.format(obj3)], param['{}_f2_A'.format(obj3)], param['{}_f3_A'.format(obj3)], param['obj3_dis_A']
        f1Bm, f2Bm, f3Bm, disBm = param['{}_f1_B'.format(obj3)], param['{}_f2_B'.format(obj3)], param['{}_f3_B'.format(obj3)], param['obj3_dis_B']
        # sample from gaussian distribution
        f1A, f2A, f3A = np.random.multivariate_normal([f1Am, f2Am, f3Am], parameters['{}cov'.format(obj3)])
        f1B, f2B, f3B = np.random.multivariate_normal([f1Bm, f2Bm, f3Bm], parameters['{}cov'.format(obj3)])
        disA, disB = np.random.normal(disAm, 20), np.random.normal(disBm, 20)

        imgAorig, imgBorig = imgA.copy(), imgB.copy()
        start, overlap, iter = 1, 0, 0
        while start or overlap:
            start = 0
            try:
                locA = random.sample(list(points_on_circle(disA/imageSize, imageSize, imageSize, x0=.5, y0=.5)),1)[0]
                imgA, imgAorig, overlap = draw_shape(f1A, f2A, f3A, shape=obj3, loc=locA, imageSize=imageSize, image=imgAorig)
                iter += 1
            except:
                iter += 1
            if iter > max_iter:  # resample features
                f1A, f2A, f3A = np.random.multivariate_normal([f1Am, f2Am, f3Am], parameters['{}cov'.format(obj3)])
                disA = np.random.normal(disAm, 20)
        start, overlap, iter = 1, 0, 0
        while start or overlap:
            start = 0
            try:
                locB = random.sample(list(points_on_circle(disB/imageSize, imageSize, imageSize, x0=.5, y0=.5)), 1)[0]
                imgB, imgBorig, overlap = draw_shape(f1B, f2B, f3B, shape=obj3, loc=locB, imageSize=imageSize, image=imgBorig)
                iter += 1
            except:
                iter += 1
            if iter > max_iter:  # resample features
                f1B, f2B, f3B = np.random.multivariate_normal([f1Bm, f2Bm, f3Bm], parameters['{}cov'.format(obj3)])
                disB = np.random.normal(disBm, 20)
        A_info['{}_f1'.format(obj3)], A_info['{}_f2'.format(obj3)],A_info['{}_f3'.format(obj3)] = f1A, f2A, f3A
        B_info['{}_f1'.format(obj3)], B_info['{}_f2'.format(obj3)], B_info['{}_f3'.format(obj3)] = f1B, f2B, f3B
        A_info['obj3_dis'] = disA
        B_info['obj3_dis'] = disB
        A_info['obj3_locx'], A_info['obj3_locy'], B_info['obj3_locx'], B_info['obj3_locy'] = list(locA) + list(locB)


        # 4. draw irrelevant object
        # print('creating obj4')
        obj4 = param['obj4']
        A_info['obj4'], B_info['obj4'] = obj4, obj4
        # get means
        f1Am, f2Am, f3Am = param['{}_f1_A'.format(obj4)], param['{}_f2_A'.format(obj4)], param['{}_f3_A'.format(obj4)]
        f1Bm, f2Bm, f3Bm = param['{}_f1_B'.format(obj4)], param['{}_f2_B'.format(obj4)], param['{}_f3_B'.format(obj4)]
        # sample from gaussian distribution
        f1A, f2A, f3A = np.random.multivariate_normal([f1Am, f2Am, f3Am], parameters['{}cov'.format(obj4)])
        f1B, f2B, f3B = np.random.multivariate_normal([f1Bm, f2Bm, f3Bm], parameters['{}cov'.format(obj4)])

        imgAorig, imgBorig = imgA.copy(), imgB.copy()
        start, overlap, iter = 1, 0, 0
        while start or overlap:
            start = 0
            locA = [np.random.uniform(0.15, .85), np.random.uniform(0.15, .85)]
            imgA, imgAorig, overlap = draw_shape(f1A, f2A, f3A, shape=obj4, loc=locA, imageSize=imageSize, image=imgAorig)
            iter += 1
            if iter > max_iter:  # resample features
                f1A, f2A, f3A = np.random.multivariate_normal([f1Am, f2Am, f3Am], parameters['{}cov'.format(obj4)])
        start, overlap, iter = 1, 0, 0
        while start or overlap:
            start = 0
            locB = [np.random.uniform(0.15, .85), np.random.uniform(0.15, .85)]
            imgB, imgBorig, overlap = draw_shape(f1B, f2B, f3B, shape=obj4, loc=locB, imageSize=imageSize, image=imgBorig)
            iter += 1
            if iter > max_iter:  # resample features
                f1B, f2B, f3B = np.random.multivariate_normal([f1Bm, f2Bm, f3Bm], parameters['{}cov'.format(obj4)])
        A_info['{}_f1'.format(obj4)], A_info['{}_f2'.format(obj4)],A_info['{}_f3'.format(obj4)] = f1A, f2A, f3A
        B_info['{}_f1'.format(obj4)], B_info['{}_f2'.format(obj4)], B_info['{}_f3'.format(obj4)] = f1B, f2B, f3B
        A_info['obj4_locx'], A_info['obj4_locy'], B_info['obj4_locx'], B_info['obj4_locy'] = list(locA) + list(locB)

        # add noise
        imgA[imgA == 0] = .5  # background 0 -> 0.5
        imgA = imgA + sigma * np.random.randn(imgA.shape[0], imgA.shape[1])

        imgB[imgB == 0] = .5  # background 0 -> 0.5
        imgB = imgB + sigma * np.random.randn(imgB.shape[0], imgB.shape[1])


        f = plt.figure()
        plt.imshow(imgA, 'gray')
        plt.axis('off')
        if j<n_train: plt.savefig('{}/train/A{}.jpg'.format(outpath,j+1), bbox_inches='tight', pad_inches=0)
        else: plt.savefig('{}/test/A{}.jpg'.format(outpath,j+1), bbox_inches='tight', pad_inches=0)
        f.clear()
        plt.close(f)
        stim_info = stim_info.append(A_info, ignore_index=True)

        f = plt.figure()
        plt.imshow(imgB, 'gray')
        plt.axis('off')
        if j<n_train: plt.savefig('{}/train/B{}.jpg'.format(outpath,j+1), bbox_inches='tight', pad_inches=0)
        else: plt.savefig('{}/test/B{}.jpg'.format(outpath,j+1), bbox_inches='tight', pad_inches=0)
        f.clear()
        plt.close(f)
        stim_info = stim_info.append(B_info, ignore_index=True)
    stim_info.to_excel('{}/stim_info.xlsx'.format(outpath), index=None)