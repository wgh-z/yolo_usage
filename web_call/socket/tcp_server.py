import socket
import cv2
import numpy as np
from ultralytics import YOLO


def decode_image(byte_data):
    nparr = np.frombuffer(byte_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def recvall(tcp: socket.socket, img_len: int):
    received = 0
    data = b''
    while img_len > 0 and received < img_len:
        chunk = tcp.recv(4096)
        data += chunk
        received += 4096
    return data

def list_to_bytes(lst):
    return bytes(str(lst), 'utf-8')

model = YOLO(r"E:\Projects\weights\yolo\v8\detect\coco\yolov8m.pt")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 2048))  # 绑定服务端地址和端口
s.listen(1)  # 监听是否有TCP连接，同时接受5个客服端的连接申请
print('waiting for connection...')
tcp, addr = s.accept()  # 接受 TCP 客户端连接，返回客户端地址和一个新的 socket 连接
print('connected with', addr)
while True:
    # data = recvall(tcp)
    img_len_str = tcp.recv(64)
    try:
        img_len = int(img_len_str.decode('utf-8').strip())
        # print('img_len', img_len)
        data = recvall(tcp, img_len)
        frame = decode_image(data)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)

        results = model.track(frame,
                        persist=True,
                    #   stream=True,
                        half=True,
                        classes = 0,
                        tracker='bytetrack.yaml',
                        verbose=False,
                        imgsz=[384, 640]
                        )

        det = results[0].boxes.data.cpu().numpy().tolist()
        det_len = len(list_to_bytes(det))
        print('len===', det_len)
        det_len_str = str(det_len).ljust(8).encode('utf-8')
        tcp.send(det_len_str)
        tcp.send(list_to_bytes(det))
    except UnicodeDecodeError:
        pass
    
tcp.close()  # 关闭TCP连接
s.close()  # 关闭socket
