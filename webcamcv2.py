# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
from datetime import datetime
import requests
from werkzeug.utils import secure_filename
import threading
import json
sys.path.append("..")


Threshold = 0.9
MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')
NUM_CLASSES = 4
second_timeout = 10


class Detector:
    def __init__(self):
        self.sess, self.detection_graph = self.setup_graph()
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.category_index = self.setup_categories()
        self.counter = {'backpack': 0, 'luggage': 0, 'bag': 0}
        self.timer = {'backpack': 0, 'luggage': 0, 'bag': 0}

    def postImg(self,url,img):
        formatDateTime = '%Y-%m-%d %H:%M:%S'
        now = datetime.now().strftime(formatDateTime)   
        data = {"image":img.tolist(),"time":now} 
        response = requests.post(url,json=data)
        res = response.json()
        print(res["Status"])
        
    @staticmethod
    def setup_graph():
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.compat.v1.Session(graph=detection_graph)

        return sess, detection_graph

    @staticmethod
    def setup_categories():
        from object_detection.utils import label_map_util
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    @staticmethod
    def line_notice():
        url = 'https://notify-api.line.me/api/notify'
        token = 'cTte0E9QNR6fPhQnAEJaRcvIVgxkOm81CvZK3K3z4FS'
        headers = {'content-type': 'application/x-www-form-urlencoded', 'Authorization': 'Bearer ' + token}
        msg = 'Alert!! Found a bag that must be suspicious'
        r = requests.post(url, headers=headers, data={'message': msg})
        print(r.text)

    def detect(self, img):
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})

        classes_ = []
        boxes_ = []

        for index, value in enumerate(scores[0]):
            if value > Threshold:
                classes_.append(self.category_index[classes[0][index]]['name'])
                boxes_.append(boxes[0][index])
                
        for idx, cls in enumerate(classes_):
            if not classes_:
                for key in self.timer.keys():
                    self.timer[key] = 0
            if cls in self.counter.keys():
                from object_detection.utils import label_map_util
                label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
                categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
                category_index = label_map_util.create_category_index(categories)
        
                self.counter[cls] += 1
                self.timer[cls] += 1
                time.sleep(1)
                print(self.timer[cls])

                vis_util.visualize_boxes_and_labels_on_image_array(
                    img,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=Threshold)
                
                if self.timer[cls] == second_timeout:
                    dateTimeObj = datetime.now()
                    timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
                    print('Found ' + cls)
                    self.line_notice()
                    print('Line notified')
                    # cv2.imwrite(os.path.join(os.getcwd(), "static", "ImgDetect",secure_filename(timestampStr + ".jpg")),img)
                    #self.postImg('http://127.0.0.1:5000/objectDetection',img)
                    print('Captured')
                    self.timer[cls] = 0
                    threading.Thread(target=self.postImg,args=["http://192.168.0.121:5000/objectDetection",img]).start()

        return {'boxes': boxes, 'scores': scores, 'classes': classes, 'num': num}


if __name__ == "__main__":
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
