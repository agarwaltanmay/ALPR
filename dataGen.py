import numpy as np
import re
import skimage.io
import matplotlib.pyplot as plt
import skimage.morphology
import skimage.color
import skimage.transform

imgs = []
jj = 0
for k in range(60):
    file = '../UFPR/training/track00' + str(k+1).zfill(2) + '/'
    for j in range(30):
        filename = file + 'track00' + str(k+1).zfill(2) + '[' + str(j+1).zfill(2) + '].'
        print(filename)
        with open(filename + 'txt', 'r', encoding="utf-8") as f:
            data = f.readlines()
        a = data[7]
        a = a[16:]
        b = re.split(' ',a)
        C= []
        for i in range(4):
             C.append(re.split(' ',b[i]))
        C = np.asarray(C).astype('int')
        print(C)
        filename = filename + 'png'
        img = skimage.img_as_float (skimage.io.imread (filename))
        im = img[C[1,0]:C[1,0]+C[3,0],C[0,0]:C[0,0]+C[2,0]]
        im = skimage.color.rgb2gray (im)
        im = skimage.transform.resize(im,(32,32))
        print(im.shape)
        imgs.append(im)

        #im = skimage.morphology.closing(im)
        # plt.imshow(im, cmap = 'gray')
        # plt.show()
        jj += 1
imgs = np.array(imgs)
print(imgs.shape)
np.save('train.npy',imgs)





