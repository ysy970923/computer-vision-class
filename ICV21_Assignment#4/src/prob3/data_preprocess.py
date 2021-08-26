from sklearn import svm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.pipeline import Pipeline
import copy
import os
from math import log
from skimage.feature import local_binary_pattern as lbp

METHOD = 'uniform'
P = 16
R = 2

train_path = '../../data/INRIAPerson/96X160H96/Train/pos/'
train_files = os.listdir(train_path)

winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins=9

padding = (0,0)
winStride = (8,8)
train_data = []
print(len(train_files))
for train_file in train_files:
    train_img = cv2.imread(train_path+train_file, cv2.IMREAD_GRAYSCALE)
    train_img = train_img[16:144,16:80] # crop
    feature = lbp(train_img, P, R, METHOD)

    feature = np.array(feature).reshape(feature.shape[0])
    feature_max, feature_min = feature.max(), feature.min()
    if feature_max == feature_min:
        continue
    feature = (feature-feature_min)/(feature_max - feature_min)
    train_data.append(feature)

with open('pos_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)

neg_train_path = '../../data/INRIAPerson/Train/neg/'
neg_files = os.listdir(neg_train_path)

padding = (0,0)
winStride = (8,8)
neg_data = []
print(len(neg_files))
for neg_file in neg_files:
    neg_img = cv2.imread(neg_train_path+neg_file, cv2.IMREAD_GRAYSCALE)
    h, w = neg_img.shape
    h_s, w_s = np.random.choice(h-128, 10), np.random.choice(w-64, 10)
    left_tops = [(h_,w_) for h_,w_ in zip(h_s, w_s)]
    for left_top in left_tops:
        img = neg_img[left_top[0]:left_top[0]+128, left_top[1]:left_top[1]+64]
        feature = lbp(img, P, R, METHOD)
        feature = np.array(feature).reshape(feature.shape[0])
        feature_max, feature_min = feature.max(), feature.min()
        if feature_max == feature_min:
            continue
        feature = (feature-feature_min)/(feature_max - feature_min)
        neg_data.append(feature)

with open('neg_train.pkl', 'wb') as f:
    pickle.dump(neg_data, f)