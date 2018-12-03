import numpy as np
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
from scipy.ndimage import interpolation as inter


def skew(img):
    thresh = skimage.filters.threshold_otsu (img)
    im = skimage.morphology.opening (img < thresh)

    def find_score(arr, angle):
        data = inter.rotate (arr, angle, reshape=False, order=0)
        hist = np.sum (data, axis=1)
        score = np.sum ((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    delta = 1
    limit = 10
    angles = np.arange (-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score (im, angle)
        scores.append (score)

    best_score = max (scores)
    best_angle = angles[scores.index (best_score)]
    print (best_angle)

    data = inter.rotate (img, best_angle, reshape=False, order=5)

    return data