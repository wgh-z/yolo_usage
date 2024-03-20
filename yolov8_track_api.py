# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import io
import cv2 as cv
import numpy as np
import torch
from flask import Flask, request
from PIL import Image
from ultralytics import YOLO


app = Flask(__name__)

DETECTION_URL = '/track/yolov8m'


@app.route(DETECTION_URL, methods=['POST'])
def predict():
    global model
    if request.method != 'POST':
        return



    if request.files.get('image'):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        persist = request.files['persist'].read().decode('utf-8')
        classes = request.files['classes'].read()
        tracker = request.files['tracker'].read().decode('utf-8')

        results = model.track(im,
                              persist=persist,
                            #   stream=True,
                              half=True,
                              classes = int(classes),
                              tracker=tracker,
                              verbose=False
                              )
        return results[0].boxes.data.cpu().numpy().tolist()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model')
    parser.add_argument('--port', default=2053, type=int, help='port number')
    # parser.add_argument('--model', nargs='+', default=['yolov5m'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    # for m in opt.model:
    model = YOLO(r"E:\Projects\weights\yolo\v8\detect\coco\yolov8m.pt")

    app.run(host='0.0.0.0', port=opt.port, debug=True)  # debug=True causes Restarting with stat
