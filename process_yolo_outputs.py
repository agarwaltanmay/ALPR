import os
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from findLetters import findLetters

def get_yolo_outputs(path="./yolo_outputs/boxes.txt"):
    f = open(path)
    lines = [line.strip() for line in f.readlines()]

    bboxes = {}
    boxes = []
    for line in lines:
        if ".jpg" in line or ".png" in line:
            if len(boxes) != 0:
                bboxes[fname] = boxes
            boxes = []

            fname = line.split('.')[0]
        else:
            corners = []
            coords = line.split(', ')
            for coord in coords:
                corners.append(coord.split(' = ')[1])
            boxes.append(corners)
    bboxes[fname] = boxes

    return bboxes


def get_detected_objects(bboxes):

    detected_objs = []
    for fname, boxes in bboxes.items():
        fname = "./yolo_outputs/" + fname + ".jpg"
        img = skimage.io.imread(fname)
        for box in boxes:
            top = int(box[0])
            left = int(box[1])
            bot = int(box[2])
            right = int(box[3])
            crop_img = img[top:bot, left:right]
            detected_objs.append(crop_img)
    detected_objs = np.array(detected_objs)

    return detected_objs
