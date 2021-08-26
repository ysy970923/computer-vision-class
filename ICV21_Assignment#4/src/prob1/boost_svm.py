import numpy as np
import cv2
import pickle

from utils import evaluate_detections
from detect import *

if __name__ == "__main__":
    # save svm classifier
    with open('svm_clf.pkl', 'rb') as f:
        clf = pickle.load(f)

    winSize = (64,128)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins=9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

    # bboxes, confidences, image_ids = run_detector_on_set(train_scn_path, train_files, clf)

    with open('train_test_result.pkl', 'rb') as f:
        test_result = pickle.load(f)
    bboxes = test_result['bboxes']
    confidences = test_result['confidences']
    image_ids = test_result['image_ids']

    label_path = '../../data/INRIAPerson/Train/labels.txt'

    gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections = evaluate_detections(np.array(bboxes), np.array(confidences), image_ids, label_path)
    fp_bboxes = np.array(bboxes)[fp]
    fp_image_ids = [image_ids[i] for i in np.nonzero(fp)[0]]
    print(len(fp_bboxes))

    fp_hog_vectors = []

    path = ''
    for image_id, bbox in zip(fp_image_ids, fp_bboxes):
        new_path = '../../data/INRIAPerson/Train/pos' + '/' + image_id
        if new_path != path:
            path = new_path
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        window = img[ymin:ymax, xmin:xmax]
        window = cv2.resize(window, (64,128), interpolation = cv2.INTER_AREA)
        hog_vector = hog.compute(window)
        hog_vector = hog_vector.reshape(hog_vector.shape[0])
        hog_max, hog_min = hog_vector.max(), hog_vector.min()
        if hog_max == hog_min:
            continue
        hog_vector = (hog_vector-hog_min)/(hog_max - hog_min)
        fp_hog_vectors.append(hog_vector)



    # clf = svm.LinearSVC(max_iter=10000)

    # get preprocessed data(hog_vectors) for train
    with open('pos_train.pkl', 'rb') as f:
        pos_data = pickle.load(f)


    with open('neg_train.pkl', 'rb') as f:
        neg_data = pickle.load(f)

    neg_data += fp_hog_vectors

    with open('neg_train.pkl', 'wb') as f:
        pickle.dump(neg_data, f)

    data = np.array(pos_data + neg_data)
    labels = np.concatenate((np.full((len(pos_data)), 1), np.full((len(neg_data)), 0)))
    print(len(data))

    # train classifier with data(hog vectors)
    clf.fit(data, labels)

    # train acc
    print("Train Accuracy:", clf.score(data, labels))

    print(np.max(clf.decision_function(neg_data)))


    # save svm classifier
    with open('svm_clf.pkl', 'wb') as f:
        pickle.dump(clf, f)
