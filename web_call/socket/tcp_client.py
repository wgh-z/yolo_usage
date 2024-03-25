# 10fps
import socket
import cv2
import sys
import time
from ultralytics.utils.plotting import Annotator, colors


names = {0: 'person', 1: 'bicycle', 10: 'fire hydrant', 11: 'stop sign',
           12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
           16: 'dog', '17': 'horse', '18': 'sheep', '19': 'cow',
           2: 'car', '20': 'elephant', '21': 'bear', '22': 'zebra',
           23: 'giraffe', '24': 'backpack', '25': 'umbrella', '26': 'handbag',
           27: 'tie', '28': 'suitcase', '29': 'frisbee', '3': 'motorcycle',
           30: 'skis', '31': 'snowboard', '32': 'sports ball', '33': 'kite',
           34: 'baseball bat', '35': 'baseball glove', '36': 'skateboard', '37': 'surfboard',
           38: 'tennis racket', '39': 'bottle', '4': 'airplane', '40': 'wine glass',
           41: 'cup', '42': 'fork', '43': 'knife', '44': 'spoon',
           45: 'bowl', '46': 'banana', '47': 'apple', '48': 'sandwich',
           49: 'orange', '5': 'bus', '50': 'broccoli', '51': 'carrot',
           52: 'hot dog', '53': 'pizza', '54': 'donut', '55': 'cake',
           56: 'chair', '57': 'couch', '58': 'potted plant', '59': 'bed',
           6: 'train', '60': 'dining table', '61': 'toilet', '62': 'tv',
           63: 'laptop', '64': 'mouse', '65': 'remote', '66': 'keyboard',
           67: 'cell phone', '68': 'microwave', '69': 'oven', '7': 'truck',
           70: 'toaster', '71': 'sink', '72': 'refrigerator', '73': 'book',
           74: 'clock', '75': 'vase', '76': 'scissors', '77': 'teddy bear',
           78: 'hair drier', '79': 'toothbrush', '8': 'boat', '9': 'traffic light'
           }

def encode_image(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, img_encoded = cv2.imencode('.jpg', img)
    return img_encoded.tobytes()

def bytes_to_list(byte_data):
    return eval(byte_data.decode('utf-8'))

def sock_client_image():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', 2048))  #服务器和客户端都在一个系统下时使用的ip和端口
    except socket.error as msg:

        print(msg)
        print(sys.exit(1))

    cap = cv2.VideoCapture(r"E:\Projects\test_data\video\MOT\MOT17\test\MOT17-01.mp4")  #打开摄像头
    d_fps = 0
    while True:
        t1 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        img = encode_image(frame)
        img_len = len(img)
        # print('img:', img_len)
        img_len_str = str(img_len).ljust(64).encode('utf-8')
        s.send(img_len_str)
        s.sendall(img)
        det_len_str = s.recv(8)
        det_len = int(det_len_str.decode('utf-8').strip())
        det_str = s.recv(det_len)
        det = bytes_to_list(det_str)

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
        cv2.putText(annotated_frame,
                    f"FPS={d_fps:.2f}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    )
        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    s.close()


if __name__ == '__main__':
    sock_client_image()