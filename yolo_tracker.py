from typing import Any
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from dependence.utils import Interpolator, Timer
from processing.regional_judgment import point_in_rect


class Track:
    def __init__(
            self,
            weight,
            imgsz=[640, 640],
            classes=[0, 2],
            tracker="bytetrack.yaml",
            verbose=False,
            show_fps=False,
            vid_stride=1
            ):
        # init params
        self.imgsz = imgsz
        self.classes = classes
        self.tracker = tracker
        self.verbose = verbose
        self.show_fps = show_fps
        # self.vid_stride = vid_stride
        # self.stride_counter = vid_stride
        # self.current_det = None
        # self.prior_det = None

        # yolo model
        self.model = YOLO(weight, task='detect')  # 35ms gpu

        # 300帧清空离场id
        self.timer = Timer(30)
        self.l_count = 15
        self.r_count = 15

        # 跳帧计算
        self.interpolator = Interpolator(vid_stride)

    def __call__(self, frame, show_id:dict, l_rate=None, r_rate=None):
        # click point
        w, h = frame.shape[1], frame.shape[0]
        l_point = (int(w * l_rate[0]), int(h * l_rate[1])) if l_rate is not None else None
        r_point = (int(w * r_rate[0]), int(h * r_rate[1])) if r_rate is not None else None
        # print('show_id==', show_id, l_point, r_point)

        # inference
        results = self.model.track(
            frame,
            persist=True,
            classes=self.classes,
            tracker=self.tracker,
            imgsz=self.imgsz,
            verbose=self.verbose
            )

        # maintain show_id
        try:
            id_set = set(results[0].boxes.id.int().cpu().tolist())
        except AttributeError:
            id_set = set()
        show_id = self.timer(id_set, show_id)

        # 自定义绘制
        annotated_frame = frame.copy()
        annotator = Annotator(annotated_frame, line_width=2, example=str(results[0].names))
        
        # 中间帧插值
        det = self.interpolator(results[0].boxes.data.cpu().numpy())
        if len(det) and len(det[0]) == 7:
            for *xyxy, id, conf, cls in reversed(det):
                c = int(cls)  # integer class
                id = int(id)  # integer id

                if l_point is not None and id not in show_id:
                    if point_in_rect(l_point, xyxy):
                        # show_id.append(id)
                        show_id = self.timer.add_delay(show_id, id)
                        l_point = None

                if r_point is not None:
                    if point_in_rect(r_point, xyxy):
                        try:
                            # show_id.remove(id)
                            del show_id[id]
                        except:
                            pass
                        r_point = None

                # 显示指定id的目标
                if id in show_id.keys() or show_id == {}:
                    label = f"{id} {results[0].names[c]} {conf:.2f}"
                    # print('xyxy', det, xyxy)
                    annotator.box_label(xyxy, label, color=colors(c, True))

        annotated_frame = annotator.result()
        return annotated_frame, show_id
