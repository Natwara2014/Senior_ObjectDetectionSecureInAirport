# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import requests
import threading
import datetime

sys.path.append("..")

formatDateTime = '%Y-%m-%d %H:%M:%S'


def postImg(url,img):
    now = datetime.datetime.now().strftime(formatDateTime)   
    data = {"image":img.tolist(),"time":now} 
    response = requests.post(url,json=data)
    res = response.json()
    print(res["Status"])

def main():
    from object_detection.utils import visualization_utils as vis_util

    print("Started")
    cv2.namedWindow("Viewer")
    webcam = cv2.VideoCapture(0)
    print("Camera accessed")
    webcam.set(3, 1280)
    webcam.set(4, 720)
    detector = Detector()
    while webcam.isOpened():
        _, frame = webcam.read()
        list_detected = detector.detect(frame)
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(list_detected['boxes']),
            np.squeeze(list_detected['classes']).astype(np.int32),
            np.squeeze(list_detected['scores']),
            detector.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=Threshold)
        cv2.imshow('Object detector', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

