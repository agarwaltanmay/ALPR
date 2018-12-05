import numpy as numpy
import process_yolo_outputs as yolo
import LPextract as LP
from net import Net
import numpy as np

if __name__ == "__main__":
    bboxes = yolo.get_yolo_outputs("./yolo_outputs/boxes.txt")
    detected_objects = yolo.get_detected_objects(bboxes)

    for im in detected_objects:
        b = LP.LPextract(im.astype(np.float))
