import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bw = skimage.color.rgb2gray (image)
    im = skimage.filters.gaussian (bw, (2, 2))
    #im = bw
    #skimage.filters.try_all_threshold(im)
    thresh = skimage.filters.threshold_otsu (im)
    #print(thresh)
    im = skimage.morphology.opening (im < thresh)
    #im = skimage.morphology.erosion (im, skimage.morphology.square (3))
    cleared = skimage.segmentation.clear_border (im)
    label_image = skimage.measure.label (cleared, background=0)
    bboxes = []
    max = np.amax (skimage.measure.regionprops (label_image)[1].area * 0.3)
    for region in skimage.measure.regionprops (label_image):
        if region.area >= max:
            bboxes.append ((region.bbox))

    box = np.array (bboxes)
    #print(box)
    bbox = box[np.argsort (box[:, 1])]


    return bbox, bw