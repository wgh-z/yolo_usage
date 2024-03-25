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
import time
from fastapi import FastAPI, Request, UploadFile
import uvicorn, json, datetime
import torch
from typing import List
from pydantic import BaseModel

# app = Flask(__name__)
app = FastAPI()

DETECTION_URL = '/track/yolov8m'

class DetectionResult(BaseModel):
    boxes: List[List[float]]

@app.post(DETECTION_URL)
async def predict(image: UploadFile = None, persist: bool = True, classes: List[int] = [], tracker: str = 'bytetrack.yaml'):
    global model

    # t1 = time.time()
    im_bytes = await image.read()
    im = Image.open(io.BytesIO(im_bytes))

    # t2 = time.time()

    results = model.track(im,
                            persist=persist,
                        #   stream=True,
                            half=True,
                            classes=classes,
                            tracker=tracker,
                            verbose=False,
                            imgsz=[384, 640]
                            )
    # t3 = time.time()
    # print(f"t2-t1={t2-t1:.2f}, t3-t2={t3-t2:.2f}")
    return DetectionResult(boxes=results[0].boxes.data.cpu().numpy().tolist())
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model')
    parser.add_argument('--port', default=2053, type=int, help='port number')
    # parser.add_argument('--model', nargs='+', default=['yolov5m'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    # for m in opt.model:
    model = YOLO(r"E:\Projects\weights\yolo\v8\detect\coco\yolov8m.pt")

    uvicorn.run(app, host='0.0.0.0', port=opt.port, workers=1)  # debug=True causes Restarting with stat
