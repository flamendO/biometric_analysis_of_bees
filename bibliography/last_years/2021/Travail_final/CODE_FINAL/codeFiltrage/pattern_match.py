# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:49:50 2021

@author: leont
"""

from matplotlib import pyplot as plt
from sklearn import cluster
import skimage.io as skio
from os.path import join
from skimage.morphology import binary_dilation, binary_erosion 
import skimage as sk
from skimage.filters import gabor, gaussian
import math
import numpy as np
import scipy.ndimage as scnd
import cv2
import time



# print(filepath2)
# img0 = skio.imread(filepath)
# img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
# img2 = img1.copy()
# template = cv2.imread(filepath2)
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# w, h = template.shape[::-1]

# # All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#         'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# compteur = 0
# for meth in methods:
#     img = img2.copy()
#     method = eval(meth)
#     compteur = compteur+1

#     # Apply template Matching
#     res = cv2.matchTemplate(img,template,method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#         bottom_right = (top_left[0] + w, top_left[1] + h)

#     cv2.rectangle(img,top_left, bottom_right, 255, 2)

#     fig = join(r'fig', str(compteur)) 
#     figure = plt.figure(compteur)
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     # cv2.imwrite(r"code\photos\res\_"+fig+".jpg",img)
#     plt.show()
    
    
    
plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

filename = r'Ruche_18_aile.jpg'
dirpath = r'..\code\photos\Ailes'
filepath = join(dirpath, filename)

filename2 = r'Ruche_5_1.jpg'
dirpath2 = r'..\code\photos\token'
filepath2 = join(dirpath2, filename2)
print(filepath2)
img_rgb = cv2.imread(filepath)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(filepath2,0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)


threshold = 1

loc = []
while (np.size(loc)==0) :
    threshold = threshold  - 0.01*threshold
    loc = np.where(res >= threshold)
    # pt = ( loc[1][-1], loc[0][-1])
    # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
print('seuil :', threshold )  


# threshold = 0.2
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    

fig1 = plt.figure(1)
plt.subplot(1, 2, 1), plt.imshow(img_rgb), plt.title("image contenant le token",  fontsize = 7)
plt.subplot(1, 2, 2), plt.imshow(template), plt.title("token cherché",  fontsize = 7)
 