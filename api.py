import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from flask import Flask, request, jsonify, redirect, flash, url_for
import json
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from count import count, preprocess
import os
from werkzeug.utils import secure_filename

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

# if GPU
# device = torch.device("cuda", 0)
#if CPU
device = torch.device("cpu")
weights = "runs/train/yolov7-custom8/weights/best.pt"
imgsz = 640
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(imgsz, s=stride)

def run(img):
    origin = img.copy()
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
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], origin.shape).round()
    return pred[0].cpu().numpy()

# UPLOAD_FOLDER = "C:\Windows"
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
file_path = "./temp.jpg"
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file from the request
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            # return redirect(request.url)
        file = request.files['file']
        img = file
        file.save(file_path)
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            # return redirect(request.url)
        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
            # file.save("C:\\" + filename)
            # print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('download_file', name=filename))
    else:
        print("Shouldn't use GET method")
        return
    # url = request.args.get('URL')
    # print(url)
    # img = Image.open(url).convert('RGB')
    # img = np.array(img)
    img = Image.open(file_path).convert('RGB')
    img = np.array(img)
    try:
        det = run(img)
        data = [[int(a) for a in x] for x in det]
        bboxes = []
        for bbox in data:
            offsetx = int(0.01*(bbox[2]-bbox[0]))
            offsety = int(0.11*(bbox[3]-bbox[1]))
            bboxes.append([int(bbox[0] - offsetx), int(bbox[1] - offsety), int(bbox[2] + offsetx), int(bbox[3] + offsety)])

        bbox = bboxes[0]
        bbox = [int(x) if x > 0 else 0 for x in bbox]
        key = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        h, w, _ = key.shape
        new_size = 480
        max_size = max(h, w)
        ratio = new_size/max_size
        new_h = round(h*ratio)
        new_w = round(w*ratio)
        key = cv2.resize(key, dsize = (new_w, new_h))

        key = preprocess(key)
        numBlackKey, numWhiteKey = count(key)

        if len(data) == 0:
            return json.dumps("No piano. Try again!")

        res = {
            "Black": numBlackKey,
            "White": numWhiteKey
        }
    except:
        res = " Piano is not scanned, try again with clear image "
    return json.dumps(res)

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8000)