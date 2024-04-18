# yolo追踪+自定义绘制+fps显示
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import time


# Load the YOLOv8 model
model = YOLO(r"E:\Projects\weights\yolo\v8\detect\export\openvino\yolov8s_openvino_model_384x640")  # 35ms gpu

# Open the video file
video_path = r"E:\Projects\test_data\video\MOT17-test\MOT17-03.mp4"
cap = cv2.VideoCapture(video_path)

d_fps = 0
# Loop through the video frames
while cap.isOpened():
    t1 = time.time()
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(
            frame,
            persist=True,
            classes=[0, 2],  # 0: person, 2: car
            # tracker="botsort.yaml",  # 12fps
            tracker="bytetrack.yaml",  # 20fps
            imgsz=(384, 640),
            # half=True,
            verbose=False,
            )

        # det = results[0].boxes.data.cpu().numpy()
        # names = results[0].names
        # boxes = results[0].boxes.xywh.cpu()
        # cls = results[0].boxes.cls.cpu()
        # conf = results[0].boxes.conf.cpu()
        # track_ids = results[0].boxes.id.int().cpu().tolist()

        # 自定义绘制
        annotated_frame = frame.copy()
        annotator = Annotator(annotated_frame, line_width=2, example=str(results[0].names))
        det = results[0].boxes.data.cpu().numpy()
        if len(det) and len(det[0]) == 7:  # 有目标，且有id元素
          for *xyxy, id, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = f"{int(id)} {results[0].names[c]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(c, True))

        annotated_frame = annotator.result()

        # 内置绘制
        # annotated_frame = results[0].plot()

        d_fps = (d_fps + (1 / (time.time() - t1))) / 2
        im0 = cv2.putText(annotated_frame, f"FPS={d_fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 显示fps

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
