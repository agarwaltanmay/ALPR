import cv2
from skimage.feature import canny
import skimage.color
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import matplotlib.patches


def LPextract(im):
    image = skimage.color.rgb2gray (im)
    edges = canny (image, 1, 1, 200)
    edges = img_as_ubyte(edges)
    _, contours, _ = cv2.findContours (edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = []
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if w > 100 or h > 100:
            box.append((y, x, y+h, x+w))
    for bbox in box:
        plt.imshow (image)
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle ((minc, minr), maxc - minc, maxr - minr,
                                             fill=False, edgecolor='red', linewidth=2)
        plt.gca ().add_patch (rect)
        plt.show ()

    LPbox = []
    return LPbox

