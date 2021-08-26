import os
import pickle
import cv2
import numpy as np
import multiprocessing as mp

def worker(block_list, base):
    print("process start", base)
    train_path = '../../data/INRIAPerson/64X128neg/'
    train_files = os.listdir(train_path)

    for i, block in enumerate(block_list):
        block_size = (block[2]-block[0], block[3]-block[1])
        winSize = block_size
        blockSize = block_size
        blockStride = (1,1)
        cellSize = (int(block_size[0]/2), int(block_size[1]/2))
        nbins=9
        neg_train = []
        for train_file in train_files:
            train_img = cv2.imread(train_path+train_file)    
            train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
            window = train_img[block[1]:block[3], block[0]:block[2]]
            hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
            hog_vector = hog.compute(window) # hog compute
            hog_vector = hog_vector.reshape(hog_vector.shape[0])
            hog_max, hog_min = hog_vector.max(), hog_vector.min()
            if hog_max == hog_min:
                continue
            hog_vector = (hog_vector-hog_min)/(hog_max - hog_min)
            neg_train.append(hog_vector)
        with open(f'cascade_data/neg/{base+i}.pkl', 'wb') as f:
            pickle.dump(neg_train, f)
        print(base+i)

if __name__ == "__main__":
    # train_path = '../../data/INRIAPerson/96X160H96/Train/pos/'
    # train_files = os.listdir(train_path)
    with open('./cascade_model/block_list.pkl', 'rb') as f:
        block_list = pickle.load(f)

    # for i, block in enumerate(block_list):
    #     print(block)
    #     block_size = (block[2]-block[0], block[3]-block[1])
    #     winSize = block_size
    #     blockSize = block_size
    #     blockStride = (1,1)
    #     cellSize = (int(block_size[0]/2), int(block_size[1]/2))
    #     nbins=9
    #     pos_train = []
    #     for train_file in train_files:
    #         train_img = cv2.imread(train_path+train_file, cv2.IMREAD_GRAYSCALE)
    #         train_img = train_img[16:144,16:80] # crop
    #         window = train_img[block[1]:block[3], block[0]:block[2]]
    #         hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    #         hog_vector = hog.compute(window,winStride,padding) # hog compute
    #         hog_vector = hog_vector.reshape(hog_vector.shape[0])
    #         hog_max, hog_min = hog_vector.max(), hog_vector.min()
    #         if hog_max == hog_min:
    #             continue
    #         hog_vector = (hog_vector-hog_min)/(hog_max - hog_min)
    #         pos_train.append(hog_vector)
    #     with open(f'./cascade_data/pos/{i}.pkl', 'wb') as f:
    #         pickle.dump(pos_train, f)

    # ## neg
    procs = []
    l = len(block_list)
    for i in range(8):
        if i == 7:
            p = mp.Process(target=worker, args=(block_list[i*int(l/8):], i*int(l/8)))
        else:
            p = mp.Process(target=worker, args=(block_list[i*int(l/8):(i+1)*int(l/8)], i*int(l/8)))
        procs.append(p)
        p.start()
            
    for p in procs:
        p.join()

    