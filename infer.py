import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

path = "test.jpeg"
device = torch.device("cuda", 0)
weights = "runs/train/yolov7-custom8/weights/best.pt"
imgsz = 640
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(imgsz, s=stride)
origin = cv2.imread(path)
img= origin.copy()
img = letterbox(img, imgsz, stride)[0]
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).to(device).float()
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if len(img.shape) == 3:
    img = img[None]
with torch.no_grad():
    pred = model(img.to(device), augment=False)[0]
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
pred = non_max_suppression(pred, conf_thres, iou_thres)
for i, det in enumerate(pred): 
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], origin.shape).round()

# for bbox in pred[0]:
#     bbox = bbox.cpu().numpy()
#     origin = cv2.rectangle(origin, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
# cv2.imshow("image", origin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

bbox = pred[0][0].cpu().numpy()
bbox = [int(x) for x in bbox]
key = origin[bbox[1]:bbox[3], bbox[0]:bbox[2]]

gray = cv2.cvtColor(key, cv2.COLOR_BGR2GRAY)

ret,thresh1 = cv2.threshold(gray,225,255,cv2.THRESH_BINARY)
tmp = np.sum(thresh1, axis=1)
tmp = (tmp - np.min(tmp)) / np.max(tmp)
idx = np.where(tmp > 0.7)[0][0]

# binary = cv2.bitwise_not(gray)
# contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# edged = cv2.Canny(thresh1, 30, 200)
# contours, hierarchy = cv2.findContours(edged, 
#     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# cv2.drawContours(key, contours, -1, (0, 255, 0), 3)
  
# cv2.imshow('Contours', key)

# key = cv2.line(key, (0, idx-1), (key.shape[1]-1, idx-1), (0, 255, 0), 1)
black_key = key[:idx, :]
white_key = key[idx:, :]

ret,white_key_thresh = cv2.threshold(white_key,225,255,cv2.THRESH_BINARY)
white_key_thresh = cv2.bitwise_not(white_key_thresh)
kernel = np.ones((3, 3), np.uint8)
white_key_thresh = cv2.dilate(white_key_thresh, kernel, iterations=1)

# contours, hierarchy = cv2.findContours(white_key_thresh.astype(np.float32), 
#     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(white_key, contours, -1, (0, 255, 0), 1)

from skimage import measure

objects = measure.label(white_key_thresh)
props_list = measure.regionprops(objects)
print(props_list)
# num_scissors = 0
# for props in props_list:  # one RegionProps object per region
#     if props.euler_number == -1:
#         num_scissors += 1

cv2.imshow("test", white_key_thresh)
# cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
