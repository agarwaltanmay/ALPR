import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
from scipy.ndimage import interpolation as inter
from findLetters import findLetters
import matplotlib.pyplot as plt
import matplotlib.patches
import torch.nn.functional as F
import torch.nn as nn
from skew import skew
from PIL import Image as im
import string

class Net(nn.Module):
    def __init__(self):
        super (Net, self).__init__ ()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "Net"

class LeNet(nn.Module):
    def __init__(self):
        super (LeNet, self).__init__ ()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 50, 500)
        self.fc2 = nn.Linear(500, 36)

    def forward(self, x):
        #x = x.view(-1,1,32,32)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"



letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = pad_value = kwargs.get('padder', 255)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

im1 = skimage.img_as_float(skimage.io.imread('test3.jpg'))
im1 = skew(im1[350:405,330:545])
plt.imshow (im1, cmap = 'gray')

plt.show()
#im1[420:460,255:388] im1
#im1[407:444,328:465] im2
#im1[350:405,330:545] im3

bboxes, bw = findLetters(im1)
l = []
for bbox in bboxes:
    plt.imshow (bw, cmap='gray')
    minr, minc, maxr, maxc = bbox
    rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                            fill=False, edgecolor='red', linewidth=2)
    plt.gca().add_patch(rect)
plt.show()

model = torch.load('model_36.pt')
for i in range (3):
        let = bw[bboxes[i, 0]:bboxes[i, 2], bboxes[i, 1]:bboxes[i, 3]]
        let = skimage.transform.resize (let, (28, 28))
        let = (np.pad (let, 2, pad_with, padder=let[0, 0]).transpose ())
        #let = 1.0 - let/ np.amax (let)
        #let = skimage.morphology.opening (let)
        plt.imshow (let, cmap='gray')

        #plt.imshow (let)
        #plt.show()
        let = torch.from_numpy(let)
        let = let.view(1,1,32,32)
        probs = model(let.float())
        pr = probs.data.numpy()
        pr = pr / np.amax (pr, axis=1)[:, np.newaxis]
        pr[pr < 1.0] = 0.0
        y = np.argmax (pr[:])
        #print(y)
        l.append (letters[y])
        #print(letters[y])
model = torch.load('model_10.pt')
letters = np.array([str(_) for _ in range(10)])
for i in range (3,bboxes.shape[0]):
        let = bw[bboxes[i, 0]-5:bboxes[i, 2]+5, bboxes[i, 1]-5:bboxes[i, 3]+5]
        let = skimage.transform.resize (let, (28, 28))
        #let = (np.pad (let, 2, pad_with, padder=1.0))
        let = 1.0 - let/ np.amax (let)
        let = skimage.morphology.erosion (let)
        #let = skimage.morphology.dilation (let)
        # thresh = skimage.filters.threshold_otsu (let)
        # print(thresh)
        # let = skimage.morphology.opening (let > thresh)
        #let = (np.pad (let, 3, pad_with, padder=0.0))
        plt.imshow (let, cmap='gray')

        #plt.imshow (let)
        plt.show()
        let = torch.from_numpy(let)
        let = let.view(1,1,28,28)
        probs = model(let.float())
        pr = probs.data.numpy()
        pr = pr / np.amax (pr, axis=1)[:, np.newaxis]
        pr[pr < 1.0] = 0.0
        y = np.argmax (pr[:])
        #print(y)
        l.append (letters[y])
        #print(letters[y])

print(l)
