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

# make my classifier
svm_clf = svm.LinearSVC(max_iter=10000, class_weight='balanced')

# get preprocessed data(hog_vectors) for train
with open('pos_train.pkl', 'rb') as f:
    pos_data = pickle.load(f)

with open('neg_train.pkl', 'rb') as f:
    neg_data = pickle.load(f)

print(len(neg_data))



data = np.array(pos_data + neg_data)
labels = np.concatenate((np.full((len(pos_data)), 1), np.full((len(neg_data)), 0)))
print(len(data))

# train classifier with data(hog vectors)
svm_clf.fit(data, labels)

# train acc
print("Train Accuracy:", svm_clf.score(data, labels))

print(np.max(svm_clf.decision_function(neg_data)))


# save svm classifier
with open('svm_clf.pkl', 'wb') as f:
    pickle.dump(svm_clf, f)