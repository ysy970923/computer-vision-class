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
import sklearn.metrics as metrics
import time
import multiprocessing as mp
import random

from dataLoader import load_data

class Cascade_train:
    def __init__(self):
        self.levels = []

    def train(self, n_blocks, pos_files, neg_files):
        max_n_levels = 30
        i = 0
        D = [1. for _ in range(max_n_levels)]
        F = [1. for _ in range(max_n_levels)]
        d_min = 0.9975
        f_max = 0.7
        F_target = 0.01
        print(len(neg_files), 'neg files left')
        while F[i] > F_target:
            if len(self.levels) > i:
                print("level", i, "already made")
                pred = self.levels[i].predict(pos_files, neg_files)
                labels = np.concatenate((np.full((len(pos_files)), 1), np.full((len(neg_files)), 0)))
                tn, fp, fn, tp = metrics.confusion_matrix(labels, pred).ravel()
                f = fp/(tn+fp)
                print('f', f)
            else: 
                adaboost = AdaBoost_train()

                f = adaboost.train(n_blocks, pos_files, neg_files, f_max, d_min)
                self.levels.append(adaboost)
                adaboost.save(f'level_{i}.pkl')

            F[i+1] = F[i]*f
            D[i+1] = D[i]*d_min
            # false_positive only left for neg set
            neg_files = neg_files[self.levels[i].predict([],neg_files) == 1]
            print("level", i, "made", len(neg_files), 'neg files left')
            i += 1

        
        print(F)
        print(D)
    
    def save(self):
        for i, level in enumerate(self.levels):
            level.save(f'level_{i}.pkl')

def svm_trainer(task_queue, ret_queue, pos_files, neg_files, sample_weight):
    print("process made")
    while True:
        block_idx = task_queue.get()
        if block_idx == -1:
            break
        sample_data, sample_labels = load_data(block_idx, pos_files, neg_files)
        # make and train SVMs
        clf = svm.LinearSVC(class_weight='balanced')
        clf.fit(sample_data, sample_labels, sample_weight)
        sample_pred = clf.predict(sample_data)
        # tn, fp, fn, tp = metrics.confusion_matrix(sample_labels, sample_pred).ravel()
        # score = tp/(tp+fn)
        score = 1 - np.dot((sample_labels != sample_pred), sample_weight)/np.sum(sample_weight)
        # score = metrics.accuracy_score(labels, pred)
        ret_queue.put({'block_idx': block_idx, 'clf':copy.deepcopy(clf), 'score':score})

class AdaBoost_train:
    def __init__(self):
        self.blocks = []
        self.block_idxs = []
        self.clfs = []
        self.alphas = []
        self.threshold = 0

    def save(self, filename):
        save_data = dict()
        save_data['blocks'] = self.blocks
        save_data['block_idxs'] = self.block_idxs
        save_data['clfs'] = self.clfs
        save_data['alphas'] = self.alphas
        save_data['threshold'] = self.threshold
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

    def train(self, n_blocks, pos_files, neg_files, f_max, d_min):
        with open('./cascade_model/block_list.pkl', 'rb') as f:
            block_list = pickle.load(f)
        sample_pos_files = pos_files
        neg_files_copy = copy.deepcopy(neg_files)
        random.shuffle(neg_files_copy)
        sample_neg_files = neg_files_copy[:min(len(neg_files), 2*len(pos_files))]
        labels = np.concatenate((np.full((len(pos_files)), 1), np.full((len(neg_files)), 0)))
        sample_weight = np.full((len(sample_pos_files)+len(sample_neg_files)), 1)
        f = 1.
        while f > f_max:
            best_clf, best_score = None, 0

            # train 5% of linear SVMs with matching blocks and find best svm
            idxs = np.random.choice(n_blocks, int(n_blocks*0.05), replace=False)
            task_queue = mp.Queue()
            ret_queue = mp.Queue()

            procs = []
            for _ in range(8):
                p = mp.Process(target=svm_trainer, args=(task_queue, ret_queue, sample_pos_files, sample_neg_files, sample_weight))
                procs.append(p)
                p.start()
            
            for idx in idxs:
                task_queue.put(idx)

            for i in range(len(idxs)):
                msg = ret_queue.get()
                score = msg['score']
                if score > best_score:
                    best_idx = msg['block_idx']
                    best_clf = msg['clf']
                    best_score = score
                    print(score)
            for _ in procs:
                task_queue.put(-1)
            
            for p in procs:
                p.join()

            self.block_idxs.append(best_idx)
            self.blocks.append(block_list[best_idx])
            self.clfs.append(best_clf)
            sample_data, sample_labels = load_data(best_idx, sample_pos_files, sample_neg_files)
            sample_pred = best_clf.predict(sample_data)
            total_error = np.dot((sample_labels != sample_pred), sample_weight)/np.sum(sample_weight)
            print(total_error)
            alpha = 1/2*log((1-total_error)/total_error)
            self.alphas.append(alpha)
            print(self.alphas)

            sample_confidences = self.predict_proba(sample_pos_files, sample_neg_files)
            print(sample_weight)
            sample_weight = np.square(sample_confidences - sample_labels)+0.1
            print(sample_weight)

            confidences = self.predict_proba(pos_files, neg_files)
            print(confidences)
            threshold = 1
            d = 0
            while d < d_min:
                threshold -= 0.003
                pred = (confidences > threshold)
                tn, fp, fn, tp = metrics.confusion_matrix(labels, pred).ravel()
                d = tp/(tp+fn)
                if d > 0.98:
                    print(fn)
            print(d)
            self.threshold = threshold
            pred = (confidences > threshold)
            tn, fp, fn, tp = metrics.confusion_matrix(labels, pred).ravel()
            f = fp/(tn+fp)
            print(f)
            print(len(self.clfs))

        return f

    
    def predict_proba(self, pos_files, neg_files):
        total_score = np.zeros(len(pos_files) + len(neg_files))
        for i in range(len(self.alphas)):
            total_data, labels = load_data(self.block_idxs[i], pos_files, neg_files)
            total_score += self.alphas[i]*self.clfs[i].predict(total_data)
        return total_score/np.sum(self.alphas)
    
    def predict(self, pos_files, neg_files):
        score = self.predict_proba(pos_files, neg_files)
        return (score > self.threshold)

if __name__ == '__main__':

    with open(f"../cascade_data/Train/pos/1.pkl", 'rb') as f:
        data = pickle.load(f)
    pos_files = np.arange(len(data))

    with open(f"../cascade_data/Train/neg1/1.pkl", 'rb') as f:
        data = pickle.load(f)
    neg_files = np.arange(len(data))

    with open(f"../cascade_data/block_list.pkl", 'rb') as f:
        data = pickle.load(f)


    n_blocks = len(data)

    cascade = Cascade_train()

    level_0 = AdaBoost_train()
    with open('level_0.pkl', 'rb') as f:
        save_data = pickle.load(f)
    print(save_data)
    level_0.blocks = save_data['blocks']
    level_0.block_idxs = save_data['block_idxs']
    level_0.clfs = save_data['clfs']
    level_0.alphas = save_data['alphas']
    level_0.threshold = save_data['threshold']
    cascade.levels.append(level_0)

    # level_1 = AdaBoost_train()
    # with open('level_1.pkl', 'rb') as f:
    #     save_data = pickle.load(f)
    # level_1.blocks = save_data['blocks']
    # level_1.block_idxs = save_data['block_idxs']
    # level_1.clfs = save_data['clfs']
    # level_1.alphas = save_data['alphas']
    # level_1.threshold = save_data['threshold']
    # cascade.levels.append(level_1)

    # level_2 = AdaBoost_train()
    # with open('level_2.pkl', 'rb') as f:
    #     save_data = pickle.load(f)
    # level_2.blocks = save_data['blocks']
    # level_2.clfs = save_data['clfs']
    # level_2.alphas = save_data['alphas']
    # level_2.threshold = save_data['threshold']
    # cascade.levels.append(level_2)

    cascade.train(n_blocks, pos_files, neg_files)

    cascade.save()