from sklearn import svm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.pipeline import Pipeline
import copy
import os

winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins=9

padding = (0,0)
winStride = (8,8)
train_data = []

train_path = '../../data/INRIAPerson/70X134H96/Train/pos/'
train_files = os.listdir(train_path)

padding = (0,0)
winStride = (8,8)
train_data = []
print(len(train_files))
for train_file in train_files:
    train_img = cv2.imread(train_path+train_file)
    train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
    for i in [3]:
        for j in [3]:
            crop_img = train_img[i:i+128,j:j+64] # crop
            # plt.imshow(crop_img)
            # plt.show()
            hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
            hog_vector = hog.compute(crop_img,winStride,padding) # hog compute
            hog_vector = np.array(hog_vector).reshape(hog_vector.shape[0])
            
            hog_max, hog_min = hog_vector.max(), hog_vector.min()
            if hog_max == hog_min:
                continue
            hog_vector = (hog_vector-hog_min)/(hog_max - hog_min)
            train_data.append(hog_vector)

print(len(train_data))
with open('pos_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)

# # neg train
neg_train_path = '../../data/INRIAPerson/Train/neg/'
neg_files = os.listdir(neg_train_path)
padding = (0,0)
winStride = (8,8)
neg_data = []
print(len(neg_files))
for neg_file in neg_files:
    neg_img = cv2.imread(neg_train_path+neg_file)
    neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB)
    h, w, _ = neg_img.shape
    h_s, w_s = np.random.choice(h-128, 10), np.random.choice(w-64, 10)
    left_tops = [(h_,w_) for h_,w_ in zip(h_s, w_s)]
    for left_top in left_tops:
        img = neg_img[left_top[0]:left_top[0]+128, left_top[1]:left_top[1]+64]
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
        hog_vector = hog.compute(img,winStride,padding)
        hog_vector = np.array(hog_vector).reshape(hog_vector.shape[0])
        hog_max, hog_min = hog_vector.max(), hog_vector.min()
        if hog_max == hog_min:
            continue
        hog_vector = (hog_vector-hog_min)/(hog_max - hog_min)
        neg_data.append(hog_vector)

print(len(neg_data))
with open('neg_train.pkl', 'wb') as f:
    pickle.dump(neg_data, f)