import torch.nn as nn
import torch.nn.functional as F
import torch
import skimage.transform
import numpy as np
import string

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = pad_value = kwargs.get('padder', 255)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def ocr(bboxes , bw):
    letters = np.array ([_ for _ in string.ascii_uppercase[:26]] + [str (_) for _ in range (10)])
    l = []
    model = torch.load ('./saved_model/model_36.pt')
    for i in range (3):
        let = bw[bboxes[i, 0]:bboxes[i, 2], bboxes[i, 1]:bboxes[i, 3]]
        let = skimage.transform.resize (let, (28, 28))
        let = (np.pad (let, 2, pad_with, padder=let[0, 0]).transpose ())
        # let = 1.0 - let/ np.amax (let)
        # let = skimage.morphology.opening (let)
        let = torch.from_numpy (let)
        let = let.view (1, 1, 32, 32)
        probs = model (let.float ())
        pr = probs.data.numpy ()
        pr = pr / np.amax (pr, axis=1)[:, np.newaxis]
        pr[pr < 1.0] = 0.0
        y = np.argmax (pr[:])
        l.append (letters[y])
    model = torch.load ('./saved_model/model_10.pt')

    letters = np.array ([str (_) for _ in range (10)])

    for i in range (3, bboxes.shape[0]):
        let = bw[bboxes[i, 0] - 5:bboxes[i, 2] + 5, bboxes[i, 1] - 5:bboxes[i, 3] + 5]
        let = skimage.transform.resize (let, (28, 28))
        # let = (np.pad (let, 2, pad_with, padder=1.0))
        let = 1.0 - let / np.amax (let)
        let = skimage.morphology.erosion (let)
        let = torch.from_numpy (let)
        let = let.view (1, 1, 28, 28)
        probs = model (let.float ())
        pr = probs.data.numpy ()
        pr = pr / np.amax (pr, axis=1)[:, np.newaxis]
        pr[pr < 1.0] = 0.0
        y = np.argmax (pr[:])
        l.append (letters[y])
    return l
