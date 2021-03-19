from flask import Flask, render_template
from flask import request,jsonify
from datetime import datetime
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import requests
import threading
from werkzeug.utils import secure_filename
import mysql.connector
import pymysql
from ObjDetectSql import *


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

sys.path.append("..")

app = Flask(__name__)
conn = pymysql.connect('localhost', 'root', '', 'seniorproject')
def object(image,datetime):
    Threshold = 0.9
    MODEL_NAME = 'inference_graph'
    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')
    NUM_CLASSES = 4
    second_timeout = 10




@app.route("/")
def hello():
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM project")
        rows = cur.fetchall()
    return render_template('web.html', datas=rows)

#API
@app.route('/objectDetection', methods=["POST"])
def objectDetection():
    print("Image has been archived")
    data = request.json
    image = np.array(data["image"],dtype="uint8")
    print('t50e -- ',data["time"])
    date,time = data["time"].split()
    #object(image,data["time"])
    # dateTimeObj = datetime.now()
    # timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    cv2.imwrite(os.path.join(os.getcwd(), "static", "ImgDetect",secure_filename(data["time"] + ".jpg")),image)
    InsertData(date, time, str(secure_filename(data["time"]+ ".jpg")))
    '''return jsonify({"Status": "success"}),200'''
    return jsonify({"Status": "success"}),200


if __name__ == "__main__":
    app.run(debug=True,threaded=True,host= '192.168.0.121')
