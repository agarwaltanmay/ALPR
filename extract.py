import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
from matplotlib import cm
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data
import cv2
from skimage import img_as_ubyte
from skew import skew
from findLetters import findLetters
from LPextract import LPextract



print(data.camera().shape)
test = data.camera()
img = skimage.img_as_float(skimage.io.imread('test3.jpg'))
im1 = img*255
l = LPextract(im1)
#
# image = skimage.color.rgb2gray (im1)
# print(image.shape)
#
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# ax = axes.ravel()
#
# edges = canny(image, 1, 1, 200)
# lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
#                                  line_gap=3)
# ax[0].imshow(image, cmap=cm.gray)
#
#
# ax[0].imshow(edges, cmap=cm.gray)
# ax[0].set_title('Canny edges')
#
# ax[1].imshow(edges * 0)
# for line in lines:
#     p0, p1 = line
#     ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
# ax[1].set_xlim((0, image.shape[1]))
# ax[1].set_ylim((image.shape[0], 0))
# ax[1].set_title('Probabilistic Hough')
# ax[1].set_axis_off()
#
# plt.tight_layout()
# plt.show()
#
#
# edges = img_as_ubyte(edges)
# #thresh = cv2.adaptiveThreshold(edge, 255, 1, 1, 11, 2)
# _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# box = []
# for cnt in contours:
#     [x, y, w, h] = cv2.boundingRect(cnt)
#     if w > 100 or h > 100:
#         [x, y, w, h] = cv2.boundingRect(cnt)
#         box.append((y, x, y+h, x+w))




# 
# bw = skimage.color.rgb2gray (im1)
# #im = skimage.filters.gaussian (im, (2, 2))
# thresh = skimage.filters.threshold_otsu (bw)
# im = skimage.morphology.opening (bw < thresh)
# 
# cleared = skimage.segmentation.clear_border (edges)
# label_image = skimage.measure.label (cleared, background=0)
# bboxes = []
# # max = np.amax (skimage.measure.regionprops (label_image)[1].area * 0.3)
# for region in skimage.measure.regionprops (label_image):
#      if region.area >= 400:
#         bboxes.append ((region.bbox))
#
# box = np.array (bboxes)

for bbox in box:
    plt.imshow (img, cmap='gray')
    minr, minc, maxr, maxc = bbox
    rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                            fill=False, edgecolor='red', linewidth=2)
    im1 = skew(img[minr:maxr, minc:maxc])
    try:
        bboxes, bw = findLetters (im1)
        plt.gca ().add_patch (rect)
        print(bboxes.shape[0])
        plt.show()
    except:
        continue
    #plt.imshow (im1, cmap='gray')
    #plt.show()
    # im1[420:460,255:388] im1
    # im1[407:444,328:465] im2
    # im1[350:405,330:545] im3

#
# 
# plt.imshow(im1)

