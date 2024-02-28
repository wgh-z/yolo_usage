from ultralytics import YOLO

model = YOLO(r'E:\Projects\weight\yolo\v8\detect\coco\yolov8s.pt')

# Export the model
model.export(
    # format='torchscript',
    # format='onnx',
    format='openvino',
    # format='engine',
    # format='coreml',
    # format='saved_model',
    # format='pb',
    # format='tflite',
    # format='edgetpu',
    # format='tfjs',
    # format='ncnn',
    imgsz=[384, 640], # image size int or (h, w) tuple
    # half=True,
    # int8=True,
    # dynamic=True,
    # simplify=True,
    # workspace=6
    )
