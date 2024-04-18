# yolo rtsp流追踪
import cv2
import time
from ultralytics import YOLO

# Load the model and run the tracker with a custom configuration file
model = YOLO(r"E:\Projects\weights\yolo\v8\detect\coco\yolov8s.pt")  # 35ms gpu

results = model.track(
    source='rtsp://192.168.31.181:8554/stream0',
    classes=[0, 2],  # 0: person, 2: car
    tracker="bytetrack.yaml",  # 20fps
    imgsz=(384, 640),
    stream=True,
    half=True,
    verbose=False
    )

for result in results:
    annotated_frame = result.plot()
    det = result.boxes.data.cpu().numpy()
    for *xyxy, id, conf, cls in reversed(det):
        frame = result.orig_img
        h, w = result.orig_shape
        print(xyxy, id, conf, cls)
