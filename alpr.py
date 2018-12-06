import numpy as numpy
import matplotlib.pyplot as plt
import process_yolo_outputs as yolo
import LPextract as LP
from net import Net
from charnet import CharNet
from numnet import NumNet
import numpy as np
from skew import skew
from findLetters import findLetters
from ocr import ocr
from scipy.ndimage import interpolation as inter

if __name__ == "__main__":
    bboxes = yolo.get_yolo_outputs("./yolo_outputs/boxes.txt")
    detected_objects = yolo.get_detected_objects(bboxes)

    for im in detected_objects:
        plt.imshow(im)
        plt.show()

        lp_img = LP.LPextract(im.astype(np.float))

        lp_img = inter.rotate(lp_img, 5, reshape=False, order=5)
        plt.imshow(lp_img, cmap='gray')
        plt.show()

        lp_img = skew(lp_img)
        plt.imshow(lp_img, cmap='gray')
        plt.show()

        cboxes, lp_bw = findLetters(lp_img)

        chars = ocr(cboxes, lp_bw)
        print(chars)

