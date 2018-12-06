import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from skimage.feature import canny
import skimage.color
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np

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
    
    saved_model = torch.load('./saved_model/trained_model.pt')
    
    min_val = 1e6
    lp_img = np.empty((10, 10))
    for bbox in box:
        minr, minc, maxr, maxc = bbox
        crop_im = image[minr:maxr, minc:maxc] / 255.0

        crop_im = skimage.transform.resize(crop_im, (32, 32))
        crop_im = torch.from_numpy(crop_im).float()
        crop_im = crop_im.view(-1, 1, 32, 32)
        probs = saved_model(crop_im)

        val = np.linalg.norm(probs.item() - 1)
        if min_val > val:
            min_val = val
            lp_img = image[minr:maxr, minc:maxc]
    
    plt.imshow(lp_img, cmap='gray')
    plt.show()

    # for bbox in box:
    #     plt.imshow (image)
    #     minr, minc, maxr, maxc = bbox
    #     rect = matplotlib.patches.Rectangle ((minc, minr), maxc - minc, maxr - minr,
    #                                          fill=False, edgecolor='red', linewidth=2)
    #     plt.gca ().add_patch (rect)
    #     plt.show ()

    return lp_img



