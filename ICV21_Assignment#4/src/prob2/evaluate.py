from utils import *
import pickle
import numpy as np
from detect import *
import os
from cascade_inference import Cascade_inference

if __name__ == "__main__":

    cascade = Cascade_inference()
    cascade.load('./cascade_model')

    test_scn_path = '../../data/INRIAPerson/Test/pos'
    test_files = os.listdir(test_scn_path)
    bboxes, confidences, image_ids = run_detector_on_set(test_scn_path, test_files, cascade)
    with open('test_result.pkl', 'wb') as f:
        pickle.dump({'bboxes':bboxes, 'confidences': confidences, 'image_ids': image_ids}, f)

    # path='../../data/INRIAPerson/Test/pos/person_085.png'
    # label_path = '../../data/INRIAPerson/Test/annotations/person_085.txt'
    # img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # cur_bboxes, cur_confidences = multi_scale_detector(img, clf, 0.2)

    # gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections = evaluate_detections(np.array(cur_bboxes), np.array(cur_confidences), label_path)