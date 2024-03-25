# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# 3.5fps
"""
Perform test request
"""

import pprint
import requests
import cv2 as cv
import time
import numpy as np
from ultralytics.utils.plotting import Annotator, colors


# DETECTION_URL = 'http://192.168.31.190:4006/track/yolov8m'
DETECTION_URL = 'http://127.0.0.1:4006/track/yolov8m'

names = {0: 'person', 1: 'bicycle', 10: 'fire hydrant', 11: 'stop sign',
           12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
           '16': 'dog', '17': 'horse', '18': 'sheep', '19': 'cow',
           '2': 'car', '20': 'elephant', '21': 'bear', '22': 'zebra',
           '23': 'giraffe', '24': 'backpack', '25': 'umbrella', '26': 'handbag',
           '27': 'tie', '28': 'suitcase', '29': 'frisbee', '3': 'motorcycle',
           '30': 'skis', '31': 'snowboard', '32': 'sports ball', '33': 'kite',
           '34': 'baseball bat', '35': 'baseball glove', '36': 'skateboard', '37': 'surfboard',
           '38': 'tennis racket', '39': 'bottle', '4': 'airplane', '40': 'wine glass',
           '41': 'cup', '42': 'fork', '43': 'knife', '44': 'spoon',
           '45': 'bowl', '46': 'banana', '47': 'apple', '48': 'sandwich',
           '49': 'orange', '5': 'bus', '50': 'broccoli', '51': 'carrot',
           '52': 'hot dog', '53': 'pizza', '54': 'donut', '55': 'cake',
           '56': 'chair', '57': 'couch', '58': 'potted plant', '59': 'bed',
           '6': 'train', '60': 'dining table', '61': 'toilet', '62': 'tv',
           '63': 'laptop', '64': 'mouse', '65': 'remote', '66': 'keyboard',
           '67': 'cell phone', '68': 'microwave', '69': 'oven', '7': 'truck',
           '70': 'toaster', '71': 'sink', '72': 'refrigerator', '73': 'book',
           '74': 'clock', '75': 'vase', '76': 'scissors', '77': 'teddy bear',
           '78': 'hair drier', '79': 'toothbrush', '8': 'boat', '9': 'traffic light'
           }

# video
cap = cv.VideoCapture(r"E:\Projects\test_data\video\MOT\MOT17\test\MOT17-01.mp4")
d_fps = 0
while True:
    t1 = time.time()
    ret, frame = cap.read()
    # Read image as OpenCV array
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image_data = cv.imencode(".jpg", img)[1].tobytes()

    det = requests.post(DETECTION_URL,
                             files={'image': image_data,
                                    # 'stream': False,  # for stream video
                                    'persist': True,  # for stream video
                                    'classes': 0,  # 0 for person
                                    'tracker': 'bytetrack.yaml'  # or 'botsort.yaml'
                                    }
                             )
    det = det.json()
    # 自定义绘制
    annotated_frame = frame.copy()
    annotator = Annotator(annotated_frame, line_width=2, example=str(names))
    if len(det) and len(det[0]) == 7:  # 有目标，且有id元素
        for *xyxy, id, conf, cls in det:
            c = int(cls)  # integer class
            label = f"{int(id)} {names[c]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(c, True))
    annotated_frame = annotator.result()
    
    d_fps = (d_fps + (1. / (time.time() - t1))) / 2
#     print(f"FPS={d_fps:.2f}")
    cv.putText(annotated_frame,
                f"FPS={d_fps:.2f}",
                (10, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                )
    cv.imshow("frame", annotated_frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv.destroyAllWindows()



# stream

# while True:
#     response = requests.post(DETECTION_URL,
#                                 files={
#                                     'stream': 'rtsp://admin:123456@192.168.31.222:554/stream1',  # for stream video
#                                     'persist': True,  # for stream video
#                                     'classes': 0,  # 0 for person
#                                     'tracker': 'bytetrack.yaml'  # or 'botsort.yaml'
#                                     }
#                                 ).json()
#     print('yes1')
#     det, im = response
#     print('yes2')
#     ims = np.numpy(im)
#     for *xyxy, id, conf, cls in det:
#         cv.rectangle(ims, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
#     # d_fps = (d_fps + (1. / (time.time() - t1+0.00001))) / 2
#     # ims = cv.putText(ims, "fps= %.2f" % (d_fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv.imshow('test', ims)
#     cv.waitKey(1)
