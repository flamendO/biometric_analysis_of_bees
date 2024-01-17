# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:10:36 2021

@author: leont
"""

# essai sans filtrage préalable, sur plusieurs ailes 

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

plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

filename = r'filtree_Ruche_18_aile_2.jpg'
dirpath = r'..\code\photos\Ailes\Ruche_18_filter'
filepath = join(dirpath, filename)

filename_token = r'Ruche_18_token_1.jpg'
dirpath_token = r'..\code\photos\token\filter\2'
filepath_token = join(dirpath_token, filename_token)

img_rgb = cv2.imread(filepath)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
template = cv2.imread(filepath_token, 0) # cv2.IMREAD_COLOR=1, cv2.IMREAD_GRAYSCALE=0

w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# ----- plusieurs patern -------
# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.15
# loc = np.where( res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
# #cv2.imwrite('res.png',img_rgb)
# ------------------------------

# -------- un seul pattern ------ identification de la meilleur méthode
# for meth in methods:
#     img = img_gray.copy()
#     method = eval(meth)
#     # Apply template Matching
#     res = cv2.matchTemplate(img,template,method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv2.rectangle(img_rgb,top_left, bottom_right, 255, 1)
#     plt.figure()
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img_rgb)
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
# ------ un seul pattern -----------

img = img_gray.copy()
method = eval('cv2.TM_SQDIFF')
# Apply template Matching
res = cv2.matchTemplate(img,template,method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_rgb,top_left, bottom_right, 2555, 2)
fig = plt.figure(1)
plt.subplot(121),plt.imshow(res) # cmap = 'gray'
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_rgb) # cmap = 'gray'
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle('cv2.TM_CCOEFF')


fig7 = plt.figure(7)
plt.subplot(1, 2, 1), plt.imshow(img_rgb), plt.title("image contenant le token",  fontsize = 7)
plt.subplot(1, 2, 2), plt.imshow(template), plt.title("token cherché",  fontsize = 7)
  

