import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.augment import LetterBox


# Load a model
model = YOLO(r'E:\Projects\weights\yolo\v8\seg\yolov8s-seg.pt')  # load an official model

# Predict with the model
results = model.predict('assets/bus.jpg')  # predict on an image
for result in results:
    # result.show()  # display results
    im0 = result.orig_img
    names = result.names
    boxes = result.boxes
    masks = result.masks
    idx = boxes.cls if boxes else range(len(masks))

    annotator = Annotator(im0, line_width=2, example=str(names))

    img = LetterBox(masks.shape[1:])(image=annotator.result())
    im_gpu = (
                torch.as_tensor(img, dtype=torch.float16, device=masks.data.device)
                .permute(2, 0, 1)
                .flip(0)
                .contiguous()
                / 255
            )
    annotator.masks(masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

    for *xyxy, conf, cls in reversed(boxes.data.cpu().numpy()):
        c = int(cls)
        label = f"{names[c]} {conf:.2f}"
        annotator.box_label(xyxy, label, color=colors(c, True))

    annotated_frame = annotator.result()
    cv2.imshow("YOLOv8 Segmentation", annotated_frame)
    cv2.waitKey(0)
