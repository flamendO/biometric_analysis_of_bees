#!/usr/bin/env python
# coding: utf-8
"""
Created on Thurs April 29 2021

@author: chloepoulic
"""

from sklearn import cluster 
import skimage.io as skio
import matplotlib.pyplot as plt 
import numpy as np
import cv2
from skimage import color
from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.color import label2rgb

#Import de l'image
plt.rcParams['image.cmap'] = 'gray' 
plt.close('all')
img_orig1 = skio.imread('/Users/chloepoulic/Documents/BZZZZ_final/Code/en_stock/Ruche_18.jpg', as_gray = True)

#Segmentation par les méthodes de K-Means, le but est de séparer les ailes du fond
img_orig = color.rgb2grey(img_orig1) 
img_orig = cv2.GaussianBlur(img_orig,(5,5), 3)

seuil = 200
img_orig = img_orig*255 > seuil
plt.figure(figsize=(15,10))

X = img_orig.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=2) 
k_means.fit(X)


clusters_means = k_means.cluster_centers_.squeeze() 
X_clustered = k_means.labels_

X_clustered.shape = img_orig.shape
fig = plt.figure(figsize=(15,10)), plt.imshow(X_clustered), plt.title('Kmean segmentation')

img = X_clustered

##Utilisation de la méthode watershed, code de la documentation https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_expand_labels.html#sphx-glr-auto-examples-segmentation-plot-expand-labels-py
# Make segmentation using edge-detection and watershed.
edges = sobel(img)

# Identify some background and foreground pixels from the intensity values.
# These pixels are used as seeds for watershed.
markers = np.zeros_like(img)
foreground, background = 1, 2
markers[img == 0] = background          #Changement par rapport au code du lien, comme on a segmenté et binarisé, on sait que le fond c'est 0
markers[img == 1] = foreground          #Changement par rapport au code du lien, comme on a segmenté et binarisé, on sait que les ailes c'est 1

ws = watershed(edges, markers)
seg1 = label(ws == foreground)

#expanded = expand_labels(seg1, distance=10)

# Show the segmentations.
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 5),
                         sharex=True, sharey=True)

color1 = label2rgb(seg1, image=img, bg_label=0)
axes[0].imshow(color1)
axes[0].set_title('Sobel+Watershed')

#color2 = label2rgb(expanded, image=coins, bg_label=0)
axes[1].imshow(edges)
axes[1].set_title('sobel')

for a in axes:
    a.axis('off')
fig.tight_layout()
plt.show()