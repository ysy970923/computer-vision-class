import pickle
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

class Cascade_inference:
    def __init__(self):
        self.levels = []

    def load(self, src_path):
        level_files = os.listdir(src_path)
        for i in range(len(level_files)-1):
            level = AdaBoost_inference()
            level.load(src_path+'/'+f'level_{i}.pkl')
            self.levels.append(level)

    def predict(self, img):
        for i, level in enumerate(self.levels[:-1]):
            if not level.predict(img):
                return False, None
        confidence = self.levels[-1].predict_proba(img)
        if confidence > self.levels[-1].threshold:
            return True, confidence
        return False, None

class AdaBoost_inference:
    def __init__(self):
        self.blocks = []
        self.clfs = []
        self.alphas = []
        self.threshold = 0
        self.hogs = []

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.blocks = data['blocks']
        self.hogs = []
        self.clfs = data['clfs']
        self.alphas = data['alphas']
        self.threshold = data['threshold']

    def predict_proba(self, gray_img):
        if len(self.hogs) == 0:
            for block in self.blocks:
                xmin, ymin, xmax, ymax = block[0], block[1], block[2], block[3]
                block_size = (xmax-xmin, ymax-ymin)
                winSize = block_size
                blockSize = block_size
                blockStride = (1,1)
                cellSize = (int(block_size[0]/2), int(block_size[1]/2))
                nbins=9
                hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
                self.hogs.append(hog)

        total_score = 0
        for i in range(len(self.alphas)):
            xmin, ymin, xmax, ymax = self.blocks[i][0], self.blocks[i][1], self.blocks[i][2], self.blocks[i][3]
            window = gray_img[ymin:ymax, xmin:xmax]
            hog_vector = self.hogs[i].compute(window)
            hog_vector = hog_vector.reshape(hog_vector.shape[0])
            total_score += self.alphas[i]*self.clfs[i].predict([hog_vector])
        return (total_score/np.sum(self.alphas))[0]
    
    def predict(self, img):
        score = self.predict_proba(img)
        return (score > self.threshold)

if __name__ == "__main__":
    cascade = Cascade_inference()
    cascade.load('./cascade_model')

    