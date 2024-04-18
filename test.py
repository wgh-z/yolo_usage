import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

coco_path = r"E:\Projects\datasets\detect\COCO\coco"
dataset_type = "train2017"
ann_file = os.path.join(coco_path, f'annotations/instances_{dataset_type}.json')
print(f'Annotation file: {ann_file}')


coco = COCO(ann_file)

img_ids = coco.getImgIds(catIds=1)  # 某一类别的图片id
print(len(img_ids), img_ids)

# for img_id in img_ids:
#     img = coco.loadImgs(img_id)[0]
#     print(img)
img_info = coco.loadImgs(img_ids[0])[0]  # 根据图片id获取图片信息
print(f'Image info: {img_info}')

imPath = os.path.join(coco_path, 'images', dataset_type, img_info['file_name'])                     
im = cv2.imread(imPath)
cv2.imshow('image', im)
cv2.waitKey(0)
