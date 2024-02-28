# yolo rtsp流追踪
from ultralytics import YOLO

# Load the model and run the tracker with a custom configuration file
model = YOLO(r"E:\Projects\weight\yolo\v8\detect\export\openvino\yolov8s_openvino_model_384x640")  # 35ms gpu

results = model.track(
    source="rtsp://admin:123456@192.168.31.211:554/stream1",
    tracker="bytetrack.yaml",  # 20fps
    imgsz=(384, 640),
    stream=True,
    verbose=False
    )

for result in results:
    det = result.boxes.data.cpu().numpy()
    for *xyxy, id, conf, cls in reversed(det):
        frame = result.orig_img
        h, w = result.orig_shape
        print(xyxy, id, conf, cls)
