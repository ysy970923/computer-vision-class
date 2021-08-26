from sklearn import svm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.pipeline import Pipeline
import copy
import multiprocessing as mp

def worker(task_queue, ret_queue, test_scn_path, clf):
    print("process made")
    while True:
        test_file = task_queue.get()
        if test_file == -1:
            break
        path=test_scn_path+'/'+test_file
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cur_image_id = test_file
        nms_threshold = 0.4
        
        cur_bboxes, cur_confidences = multi_scale_detector(img, clf, nms_threshold)
        cur_image_ids = [cur_image_id for _ in cur_bboxes]
        ret_queue.put({'cur_image_ids':cur_image_ids, 'cur_bboxes':cur_bboxes, 'cur_confidences':cur_confidences})

def run_detector_on_set(scn_path, test_files, clf):
    bboxes = []
    confidences = []
    image_ids = []
    
    task_queue = mp.Queue()
    ret_queue = mp.Queue()
    procs = []
    for _ in range(8):
        p = mp.Process(target=worker, args=(task_queue, ret_queue, scn_path, clf))
        procs.append(p)
        p.start()

    for test_file in test_files:
        task_queue.put(test_file)
    
    for _ in range(len(test_files)):
        msg = ret_queue.get()
        cur_image_ids = msg['cur_image_ids']
        cur_bboxes = msg['cur_bboxes']
        cur_confidences = msg['cur_confidences']
        bboxes += cur_bboxes
        confidences += cur_confidences
        image_ids += cur_image_ids

    for _ in procs:
        task_queue.put(-1)
            
    for p in procs:
        p.join()
        
    return bboxes, confidences, image_ids

def multi_scale_detector(img, clf, nms_threshold, window_size=(128,64)):
    # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = img
    detected = []
    confidences = []

    # scales = np.arange(0.7,10,0.3)
    scales = [1.2**i for i in range(10)]
    for scale in scales:
        gray_img1 = cv2.resize(gray_img, None, fx=1/scale, fy=1/scale, interpolation = cv2.INTER_AREA)
        if gray_img1.shape[0] < 128 or gray_img1.shape[1] < 64:
            print("too much scale")
            break
        points, confidence = sliding_window(gray_img1, clf, window_size)
        confidences += confidence
        for (x,y,x_,y_) in points:
            (x,y,x_,y_) = (int(scale*x),int(scale*y), int(scale*x_),int(scale*y_))
            detected.append((x,y,x_,y_))
    keep = nms(detected, confidences, nms_threshold)
    # print(confidences, keep, [confidences[k] for k in np.nonzero(keep)[0]])
    return [detected[k] for k in np.nonzero(keep)[0]], [confidences[k] for k in np.nonzero(keep)[0]]
    # return detected, confidences

def sliding_window(gray_img, clf, window_size = (128,64)):
    winSize = (64,128)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins=9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    h, w, _ = gray_img.shape
    h_w, w_w = window_size
    detected_boxes = []
    confidences = []
    confidence_threshold = 0
    n_h = int((h-h_w)/4)
    n_w = int((w-w_w)/2)
    for i in range(n_h):
        for j in range(n_w):
            y, x = i*4, j*2
            window = gray_img[y:y+h_w, x:x+w_w]
            # plt.imshow(window)
            # plt.show()
            hog_vector = hog.compute(window)
            hog_vector = hog_vector.reshape(hog_vector.shape[0])
            hog_max, hog_min = hog_vector.max(), hog_vector.min()
            if hog_max == hog_min:
                continue
            hog_vector = (hog_vector-hog_min)/(hog_max - hog_min)
            confidence = clf.decision_function([hog_vector])[0]
            # confidence = clf.predict_proba([hog_vector])
            # prob = np.dot(h.transpose(),clf.coef_.transpose())+clf.intercept_
            if confidence > confidence_threshold:
                detected_boxes.append(((x,y,x+w_w,y+h_w)))
                confidences.append(confidence)
    return detected_boxes, confidences

def iou(box1, box2):
  lr = min(box1[2], box2[2]) - max(box1[0], box2[0])
  if lr > 0:
    tb = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if tb > 0:
      intersection = tb*lr
      box1_size = (box1[2]-box1[0])*(box1[3]-box1[1])
      box2_size = (box2[2]-box2[0])*(box2[3]-box2[1])
      union = box1_size + box2_size - intersection
      return intersection/union

  return 0

def nms(boxes, confidences, threshold):
    confidences = np.array(confidences)
    order = confidences.argsort()[::-1]
    # print(confidences, order)
    keep = [True]*len(order)

    for i in range(len(order)-1):
        thre = threshold
        # if keep[order[i]] == False:
        #     thre *= 2
        for j in range(i+1, len(order)):
            ov = iou(boxes[order[i]], boxes[order[j]])
            if ov > thre:
                keep[order[j]] = False
    return keep