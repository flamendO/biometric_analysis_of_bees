# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:08:23 2020

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

img = skio.imread(filepath)
t0 = time.time()
img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = scnd.filters.gaussian_filter(img, sigma = 2.5)
# img = sk.img_as_float(img)
img = cv2.threshold(img0,150,255,cv2.THRESH_TRUNC)[1]

# img = cv2.adaptiveThreshold(img, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# img = scnd.filters.gaussian_filter(img, sigma = 1.1)
# img = sk.img_as_float(img)

# print("Loaded image has dimensions:", img.shape)
fig1 = plt.figure(1)
plt.subplot(1, 2, 1), plt.imshow(img0), plt.title('image originale') # cmap='gray'
plt.subplot(1, 2, 2), plt.imshow(img), plt.title('image originale thresh') # cmap='gray'

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
# X_clustered = np.uint8(X_clustered)

kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(6,6)) 
# X_clustered1 = cv2.morphologyEx(X_clustered, cv2.MORPH_OPEN, kernel)
# X_clustered2 = cv2.morphologyEx(X_clustered, cv2.MORPH_CLOSE, kernel)
# X_clustered3 = cv2.morphologyEx(X_clustered, cv2.MORPH_TOPHAT, kernel)
# X_clustered = cv2.morphologyEx(X_clustered, cv2.MORPH_BLACKHAT, kernel)
# X_clustered5 = binary_erosion(X_clustered)
# X_clustered6 = binary_dilation(X_clustered)
tf = time.time()
print("temps écoulé:", tf-t0)
fig2 = plt.figure(2)
plt.subplot(1, 1, 1), plt.imshow(X_clustered), plt.title('image culsters') # cmap='gray'

fig3 = plt.figure(3)
plt.subplot(1, 1, 1), plt.imshow(X_clustered_mod0), plt.title('image culsters modifiee 0') 

fig4 = plt.figure(4)
plt.subplot(1, 1, 1), plt.imshow(X_clustered_mod1), plt.title('image culsters modifiee 1') 

fig5 = plt.figure(5)
plt.subplot(1, 1, 1), plt.imshow(X_clustered_mod2), plt.title('image culsters modifiee 2') 




# # plt.subplot(2, 2, 1), plt.imshow(X_clustered), plt.title('Kmean segmentation OPEN')
# plt.subplot(2, 2, 2), plt.imshow(X_clustered2), plt.title('Kmean segmentation CLOSE')
# plt.subplot(2, 2, 3), plt.imshow(X_clustered3), plt.title('Kmean segmentation TOPHAT')
# plt.subplot(2, 2, 4), plt.imshow(X_clustered), plt.title('Kmean segmentation BLACKHAT')
# plt.subplot(4, 2, 5), plt.imshow(X_clustered5), plt.title('Kmean segmentation erosion')
# plt.subplot(4, 2, 6), plt.imshow(X_clustered6), plt.title('Kmean segmentation dilatation')


# # #offset = 1
# fig3 = plt.figure(3)
# for k in range(0,8):

    
#     Gx = gabor(img,0.6, k*np.pi/8, 1)[0]
#     Gy = gabor(img,0.6, k*np.pi/8, 1)[1]


#     mag = np.zeros(np.shape(img))

#     for i in range(np.shape(img)[0]):
#         for j in range(np.shape(img)[1]):
#             mag[i,j] = math.sqrt(Gx[i,j]**2+Gy[i,j]**2)
        
#     ang = np.zeros(np.shape(img)) 
        
#     for i in range(np.shape(img)[0]):
#         for j in range(np.shape(img)[1]):
#             if (Gx[i,j]==0):
#                 ang[i,j] = np.pi/2
#             else:
#                 ang[i,j] = math.atan(Gy[i,j]/Gx[i,j])
            
#     mag = gaussian(mag, sigma = 2.5)


#     # k-means clustering of the image
#     X = mag.reshape((-1, 1))
#     k_means = cluster.KMeans(n_clusters=2)
#     k_means.fit(X)

#     # extract means of each cluster & clustered population
#     clusters_means = k_means.cluster_centers_.squeeze()
#     X_clustered = k_means.labels_

#     # Display the clustered image
#     X_clustered.shape = mag.shape
#     X_clustered = binary_dilation(X_clustered)
#     X_clustered = binary_dilation(X_clustered)
    
#     #plt.subplot(3, 3, k+1), plt.imshow(mag), plt.title("magnitude ", fontsize = 7)
#     plt.subplot(3, 3, k+1), plt.imshow(X_clustered), plt.title("image cleusterisee", fontsize = 7)


# # meilleur angle:
# Gx = gabor(img,0.6, 7*np.pi/8, 1)[0]
# Gy = gabor(img,0.6, 7*np.pi/8, 1)[1]


# mag = np.zeros(np.shape(img))

# for i in range(np.shape(img)[0]):
#     for j in range(np.shape(img)[1]):
#         mag[i,j] = math.sqrt(Gx[i,j]**2+Gy[i,j]**2)
        
# ang = np.zeros(np.shape(img)) 
        
# for i in range(np.shape(img)[0]):
#     for j in range(np.shape(img)[1]):
#         if (Gx[i,j]==0):
#             ang[i,j] = np.pi/2
#         else:
#             ang[i,j] = math.atan(Gy[i,j]/Gx[i,j])
            
# mag = gaussian(mag, sigma = 2)

# X = mag.reshape((-1, 1))
# k_means = cluster.KMeans(n_clusters=2)
# k_means.fit(X)

# # extract means of each cluster & clustered population
# clusters_means = k_means.cluster_centers_.squeeze()
# X_clustered = k_means.labels_

# # Display the clustered image
# X_clustered.shape = mag.shape
# X_clustered = binary_dilation(X_clustered)
# X_clustered = binary_dilation(X_clustered)
# tf = time.time()
   
# fig4 = plt.figure(4)
# plt.subplot(1, 2, 1), plt.imshow(mag), plt.title("magnitude ", fontsize = 7)
# plt.subplot(1, 2, 2), plt.imshow(X_clustered), plt.title("image cleusterisee", fontsize = 7)

# print("temps écoulé:", tf-t0)




