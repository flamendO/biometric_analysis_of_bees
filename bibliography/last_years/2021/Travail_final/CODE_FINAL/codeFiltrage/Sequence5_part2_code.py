# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:08:23 2020

@author: capliera
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
import time  
import cv2

plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

filename = r'Aile.jpg'
dirpath = r'..\code\photos'
filepath = join(dirpath, filename)

img = skio.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

t0 = time.time()
print("Loaded image has dimensions:", img.shape)
fig1 = plt.figure(1), plt.imshow(img, cmap='gray')

# k-means clustering of the image
X = img.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=2)
k_means.fit(X)

# extract means of each cluster & clustered population
clusters_means = k_means.cluster_centers_.squeeze()
X_clustered = k_means.labels_
print('# of Observations:', X.shape)
print('Clusters Means:', clusters_means)

# Display the clustered image
X_clustered.shape = img.shape
#X_clustered = binary_erosion(X_clustered)
#X_clustered = binary_dilation(X_clustered)
fig2 = plt.figure(2), plt.imshow(X_clustered), plt.title('Kmean segmentation')

#offset = 1
fig3 = plt.figure(3)
for k in range(0,8):

    
    Gx = gabor(img,0.6, k*np.pi/8, 1)[0]
    Gy = gabor(img,0.6, k*np.pi/8, 1)[1]


    mag = np.zeros(np.shape(img))

    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            mag[i,j] = math.sqrt(Gx[i,j]**2+Gy[i,j]**2)
        
    ang = np.zeros(np.shape(img)) 
        
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if (Gx[i,j]==0):
                ang[i,j] = np.pi/2
            else:
                ang[i,j] = math.atan(Gy[i,j]/Gx[i,j])
            
    mag = gaussian(mag, sigma = 2.5)


    # k-means clustering of the image
    X = mag.reshape((-1, 1))
    k_means = cluster.KMeans(n_clusters=2)
    k_means.fit(X)

    # extract means of each cluster & clustered population
    clusters_means = k_means.cluster_centers_.squeeze()
    X_clustered = k_means.labels_

    # Display the clustered image
    X_clustered.shape = mag.shape
    X_clustered = binary_dilation(X_clustered)
    X_clustered = binary_dilation(X_clustered)
    
    #plt.subplot(3, 3, k+1), plt.imshow(mag), plt.title("magnitude ", fontsize = 7)
    plt.subplot(3, 3, k+1), plt.imshow(X_clustered), plt.title("image cleusterisee", fontsize = 7)


# meilleur angle:
Gx = gabor(img,0.6, 7*np.pi/8, 1)[0]
Gy = gabor(img,0.6, 7*np.pi/8, 1)[1]


mag = np.zeros(np.shape(img))

for i in range(np.shape(img)[0]):
    for j in range(np.shape(img)[1]):
        mag[i,j] = math.sqrt(Gx[i,j]**2+Gy[i,j]**2)
        
ang = np.zeros(np.shape(img)) 
        
for i in range(np.shape(img)[0]):
    for j in range(np.shape(img)[1]):
        if (Gx[i,j]==0):
            ang[i,j] = np.pi/2
        else:
            ang[i,j] = math.atan(Gy[i,j]/Gx[i,j])
            
mag = gaussian(mag, sigma = 2)

X = mag.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=2)
k_means.fit(X)

# extract means of each cluster & clustered population
clusters_means = k_means.cluster_centers_.squeeze()
X_clustered = k_means.labels_

# Display the clustered image
X_clustered.shape = mag.shape
X_clustered = binary_dilation(X_clustered)
X_clustered = binary_dilation(X_clustered)
tf = time.time()
   
fig4 = plt.figure(4)
plt.subplot(1, 2, 1), plt.imshow(mag), plt.title("magnitude ", fontsize = 7)
plt.subplot(1, 2, 2), plt.imshow(X_clustered), plt.title("image cleusterisee", fontsize = 7)

print('temps écoulé', tf-t0)

## algo kmean pour chaque image on initialise le centre des deux clusters
#
#
## INITIALISATION 
#
#image_bis = np.zeros(img.shape) # matrice dont les pixels valent 0 si dans cluster 0 et 1 sinon
#
#centre1 = mag[-1:-1] # initialisation des centres des clusters
#centre0 = mag[0,0]
##centre1 =mag[np.floor(mag.shape[0]/2),np.floor(mag.shape[1]/2)]
# 
#for k in range(3):
#
#    # on parcourt l image de gabor
#    for i in range(mag.shape[0]):
#        for j in range(mag.shape[1]):
#        
#            # on estime l appartenance du pixel
#            min0 = (mag[i,j]-centre0)**2
#            min1 = (mag[i,j]-centre1)**2
#            image_bis[i,j] = (min1<min0) # si le pixel appartient au cluster 1 alors min1<min0 donc on traduit cette appartenance en mettant le pixel de image_bis a 1
#        
#        # une fois l appartenance estimée, on calcule la nouvelle varience
#    var= 0
#    for i in range(image_bis.shape[0]):
#        for j in range(image_bis.shape[1]): 
#            if (image_bis[i,j] == 1):    
#                var = var + (mag[i,j]-centre1)**2
#            else: 
#                var = var + (mag[i,j]-centre0)**2
#        
#    Variance = var 
#
#     # centroid estimation update pour la prochaine itération        
#     cluster0 = 0
#     compteur0 = 0
#     cluster1 = 0
#     compteur1 = 0
#
#    for i in range(image_bis.shape[0]):
#        for j in range(image_bis.shape[1]): 
#            if (image_bis[i,j] == 1):
#                cluster1 = cluster1 + mag[i,j]
#                compteur1 = compteur1 +1
#            else:
#                cluster0 = cluster0 + mag[i,j]
#                compteur0 = compteur0 +1
#            
#    centre1 = cluster1/compteur1  
#    centre0 = cluster0/compteur0
#        
## on print l image finale et celle qu'on a avec la binarisation sans l algo kmean
#    

