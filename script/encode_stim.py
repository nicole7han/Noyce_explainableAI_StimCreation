import numpy as np
import pandas as pd
import os, glob, shutil
import os.path as op
import json
import cv2
import base64
from os import listdir
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

from maskrcnn_benchmark.structures.tsv_file_ops import *
# from tsv_file_ops import tsv_reader, tsv_writer
# from tsv_file_ops import generate_linelist_file
# from tsv_file_ops import generate_hw_file
# from maskrcnn_benchmark.structures.tsv_file import TSVFile
# from maskrcnn_benchmark.data.datasets.utils.image_ops import img_from_base64
resnet = models.resnet50(pretrained=True)
# resnet = models.resnet18(pretrained=True)
modules=list(resnet.children())[:-1]
resnet=nn.Sequential(*modules)
for p in resnet.parameters():
    p.requires_grad = False

from natsort import natsorted

pos_feat = np.hstack(([0, 0, 1, 1], [1, 1])).astype(np.float32)

fols = glob.glob('Stim_MultipleRules/Rules*')
for fol in fols: # train rules and eval rules
    img_cond = ['train','test'] if 'train' in fol else ['test'] # only training rules have both train and test images, evaluating rules have only test image
    for cond in img_cond: # combining all rules for training/testing dataset
        print('encoding {} condition'.format(cond))
        labels = []
        captions = []
        features = []
        stim_path = glob.glob('{}/*'.format(fol))
        stim_path = natsorted(stim_path)
        for rule in stim_path:
            print('creating rule {}'.format(rule))
            stim_info = pd.read_excel('{}/stim_info_text_param.xlsx'.format(rule))
            imgs = glob.glob('{}/{}/*'.format(rule, cond))
            imgs = natsorted(imgs)
            for img in imgs:
                # img = row['stim']
                # box_ids, scores, bboxes = frcnn(x)
                # ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)
                # plt.show()
                imgname = os.path.split(img)[-1].split('.jpg')[0]
                row = stim_info[stim_info['stim'] == imgname]
                image = Image.open(img)
                text = row['text'].item()

                # class label
                cls = 'A' if 'A' in imgname else 'B'
                lbl = []
                lbl.append({"class": "{}".format(cls), "rect": [0.0, 0.0, 369, 369], "conf": 1.00})
                row_label = [imgname, json.dumps(lbl)]

                # image feature encoded with resnet50
                image = preprocess(image)
                image = torch.unsqueeze(image, 0)
                img_feat = resnet(image).flatten()
                img_feat = np.hstack((img_feat, pos_feat)).astype(np.float32)
                img_feat = base64.b64encode(img_feat).decode("utf-8")
                row_feature = [imgname, json.dumps({"num_boxes":1,"features": img_feat})]

                labels.append(row_label)
                captions.append({"image_id": "{}".format(imgname), "caption": text})
                features.append(row_feature)

        # generate tsv label and feature files
        tsv_writer(labels, 'Stim_MultipleRules/{}.label.tsv'.format(cond))
        tsv_writer(features, 'Stim_MultipleRules/{}.feature.tsv'.format(cond))

        # generate linelist file
        generate_linelist_file('Stim_MultipleRules/{}.label.tsv'.format(cond), save_file='Stim_MultipleRules/{}.linelist.tsv'.format(cond))

        # generate caption json file
        with open('Stim_MultipleRules/{}_caption.json'.format(cond), 'w') as fp:
            json.dump(captions, fp)

