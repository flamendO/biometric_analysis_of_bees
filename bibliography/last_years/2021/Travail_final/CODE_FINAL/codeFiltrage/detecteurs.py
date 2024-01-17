# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:52:49 2021

@author: leont
"""

from sklearn import cluster
import skimage.io as skio
from os.path import join
import matplotlib.pyplot as plt
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

filename = r'Aile.jpg'
dirpath = r'..\code\photos'
filepath = join(dirpath, filename)

img_colors = skio.imread(filepath)
img = img_colors
t0 = time.time()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = scnd.filters.gaussian_filter(img, sigma = 2.5)
# img = sk.img_as_float(img)
img = cv2.threshold(img,150,255,cv2.THRESH_TRUNC)[1]

# img = cv2.adaptiveThreshold(img, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# img = scnd.filters.gaussian_filter(img, sigma = 1.1)
# img = sk.img_as_float(img)

# # print("Loaded image has dimensions:", img.shape)
# fig1 = plt.figure(1)
# plt.subplot(1, 2, 1), plt.imshow(img_colors), plt.title('image originale') # cmap='gray'
# plt.subplot(1, 2, 2), plt.imshow(img), plt.title('image originale thresh') # cmap='gray'

# k-means clustering of the image
X = img.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X)

# extract means of each cluster & clustered population
clusters_means = k_means.cluster_centers_.squeeze()
X_clustered = k_means.labels_
# print('# of Observations:', X.shape)
# print('Clusters Means:', clusters_means)

# Display the clustered image
X_clustered.shape = img.shape
X_clustered_mod0 = X_clustered==0
X_clustered_mod1 = X_clustered==1
X_clustered_mod2 = X_clustered==2

aile0 = X_clustered_mod0
aile1 = X_clustered_mod1
aile2 = X_clustered_mod2
aile0 = aile0.astype(np.uint8)
aile1 = aile1.astype(np.uint8)
aile2 = aile2.astype(np.uint8)

fig2 = plt.figure(2)
plt.imshow(aile0), plt.title('aile filtrée') 


# dst = cv2.cornerHarris(aile,2,3,0.04)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# img_colors[dst>0.01*dst.max()]=[0,0,255]
# tf = time.time()
# fig2 = plt.figure(2)
# plt.imshow(img_colors), plt.title('detection de points d intêret')
# print('temps écoulé', tf-t0)


# img_colors = skio.imread(filepath)
# img = cv2.cvtColor(img_colors, cv2.COLOR_BGR2GRAY)
# aile = cv2.Canny(aile, 120, 200)

# fig1 = plt.figure(1)
# plt.subplot(1, 2, 1), plt.imshow(img_colors), plt.title('image originale') # cmap='gray'
# plt.subplot(1, 2, 2), plt.imshow(img), plt.title('image Canny') # cmap='gray'

lsd1 = cv2.createLineSegmentDetector(0) 
lsd2 = cv2.createLineSegmentDetector(1)

# lines10 = lsd1.detect(aile0)[0]
# lines20 = lsd2.detect(aile0)[0]

# lines11 = lsd1.detect(aile1)[0]
# lines21 = lsd2.detect(aile1)[0]

# lines12 = lsd1.detect(aile2)[0]
# lines22 = lsd2.detect(aile2)[0]

# lsd1 = cv2.createLineSegmentDetector(0) 
# lsd2 = cv2.createLineSegmentDetector(1)

lines10 = lsd1.detect(img)[0]
lines20 = lsd2.detect(img)[0]

drawnLines10 = lsd1.drawSegments(img_colors, lines10)
drawnLines20 = lsd2.drawSegments(img_colors, lines20)

# drawnLines11 = lsd1.drawSegments(img_colors, lines11)
# drawnLines21 = lsd2.drawSegments(img_colors, lines21)

# drawnLines12 = lsd1.drawSegments(img_colors, lines12)
# drawnLines22 = lsd2.drawSegments(img_colors, lines22)

fig3 = plt.figure(3)
plt.subplot(1, 2, 1), plt.imshow(drawnLines10), plt.title('lignes 1') # cmap='gray'
plt.subplot(1, 2, 2), plt.imshow(drawnLines20), plt.title('lignes 2') # cmap='gray'

# fig4 = plt.figure(4)
# plt.subplot(1, 2, 1), plt.imshow(drawnLines11), plt.title('lignes 1') # cmap='gray'
# plt.subplot(1, 2, 2), plt.imshow(drawnLines21), plt.title('lignes 2') # cmap='gray'

# fig5 = plt.figure(5)
# plt.subplot(1, 2, 1), plt.imshow(drawnLines12), plt.title('lignes 1') # cmap='gray'
# plt.subplot(1, 2, 2), plt.imshow(drawnLines22), plt.title('lignes 2') # cmap='gray'



