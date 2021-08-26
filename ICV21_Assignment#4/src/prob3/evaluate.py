from utils import *
import pickle
import numpy as np
from detect import *
import os

if __name__ == "__main__":

    with open('svm_clf.pkl', 'rb') as f:
        clf = pickle.load(f)

    test_scn_path = '../../data/INRIAPerson/Test/pos'
    test_files = os.listdir(test_scn_path)
    bboxes, confidences, image_ids = run_detector_on_set(test_scn_path, test_files, clf)
    with open('test_result.pkl', 'wb') as f:
        pickle.dump({'bboxes':bboxes, 'confidences': confidences, 'image_ids': image_ids}, f)