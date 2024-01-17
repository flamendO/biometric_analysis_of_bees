# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:49:26 2021

@author: leont
"""

import numpy as np
import skimage.data as skd
import skimage.io as skio
import matplotlib.pyplot as plt
import copy
import skimage.exposure as ske
from os.path import join
from sklearn import cluster
from skimage.filters import gabor, gaussian
import cv2
import sklearn
from scipy import signal
import scipy.ndimage as scnd
import math 
from skimage.morphology import opening

#All figures closing
plt.close('all')

# matplotlib for gray level images display: fix the colormap and 
# the image figure size
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams["figure.figsize"] = (12,4)

# chargement de l'image RGB
filename = r'Ruche_5.jpg'
dirpath = r'..\code\photos'
filepath = join(dirpath, filename)
image_RGB = skio.imread(filepath) 
image_gray = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2GRAY)

# Algorythme de Canny == extraction des contours

# # Etape 1 : filtre gaussien

# image_gray = scnd.filters.gaussian_filter(image_gray, sigma = 2)
image_gray = cv2.bilateralFilter(image_gray, 10, 90, 90)

# Etape 2 

sobel = np.array([[ -1-1j, 0-1j,  1 -1j],

                    [-1+0j, 0+ 0j, 1 +0j],

                    [ -1+1j, 0+1j,  +1 +1j]]) # Gx + j*Gy

# sobel = np.array([[ -1-3j, 0-2j,  1 -1j],

#                    [-2+0j, 0+ 0j, 2 +0j],

#                    [ -1+1j, 0+2j,  +1 +1j]]) # Gx + j*Gy

Gx = signal.convolve2d(image_gray, sobel.real, boundary='symm', mode='same')
Gy = signal.convolve2d(image_gray, sobel.imag, boundary='symm', mode='same')

mag = np.zeros(np.shape(image_gray))

for i in range(np.shape(image_gray)[0]):
    for j in range(np.shape(image_gray)[1]):
        mag[i,j] = math.sqrt(Gx[i,j]**2+Gy[i,j]**2)
        
ang = np.zeros(np.shape(image_gray)) 

for i in range(np.shape(image_gray)[0]):
    for j in range(np.shape(image_gray)[1]):
        if (Gx[i,j]==0):
            ang[i,j] = np.pi/2
        else:
            ang[i,j] = math.atan(Gy[i,j]/Gx[i,j])
            
# Etape 3 
nouvelle_image = np.zeros(np.shape(mag))

for i in range(1, np.shape(image_gray)[0]-1):
    
    for j in range(1, np.shape(image_gray)[1]-1):
        
        phase = ang[i,j] 
        
        if (( phase >= -np.pi/6)and( phase <= np.pi/6 )): 
            
           matrice_ligne = mag[i, j-1:j+2 ]
           maximum = matrice_ligne.max()
           
           if (mag[i,j]==maximum):
                nouvelle_image[i,j] = maximum
           else:
               nouvelle_image[i,j] = 0
               
        elif ((phase > np.pi/6) and (phase < np.pi/3)): 
            
            matrice_diago = np.array([mag[i,j], mag[i+1,j-1], mag[i-1,j+1]])
            maximum = matrice_diago.max()
            
            if (mag[i,j]==maximum):
                nouvelle_image[i,j] = maximum
            else:
                nouvelle_image[i,j] = 0
           
        elif ((phase >= np.pi/3) or (phase <= -np.pi/3)):
            matrice_ligne = mag[i-1:i+2, j ]
            maximum = matrice_ligne.max()
            
            if (mag[i,j]==maximum):
                nouvelle_image[i,j] = maximum
            else:
                nouvelle_image[i,j] = 0            
            
        elif (phase < -np.pi/3)and(phase > -np.pi/6):
            matrice_diago = np.array([mag[i,j], mag[i+1,j+1], mag[i-1,j-1]])
            maximum = matrice_diago.max()
            
            if (mag[i,j]==maximum):
                nouvelle_image[i,j] = maximum
            else:
                nouvelle_image[i,j] = 0   
                
# Etape 4 selection des pixels 
m_muscle = mag.max()
Low = 0.09;
High = 0.2;
Tlow = Low * m_muscle
Thigh = High * m_muscle
img_binaire1 = 3*np.ones(nouvelle_image.shape, dtype = np.uint8)

for i in range(nouvelle_image.shape[0]):
    for j in range(nouvelle_image.shape[1]):
        if (nouvelle_image[i,j]<Tlow):
            img_binaire1[i,j] = 0
        elif (nouvelle_image[i,j]>Thigh):
            img_binaire1[i,j] = 1
            
 # test pour les voisins
img_binaire = np.copy(img_binaire1) 
img_binaire[0,:] = 0
img_binaire[nouvelle_image.shape[0]-1,:] = 0
img_binaire[:,0] = 0
img_binaire[:,image_gray.shape[1]-1] = 0

for i in range(1,nouvelle_image.shape[0]-1):
    for j in range(1,nouvelle_image.shape[1]-1):
        if (img_binaire[i,j] == 3):
            if ((img_binaire[i+1,j]==1)or(img_binaire[i-1,j]==1)or(img_binaire[i,j+1]==1)or(img_binaire[i,j-1]==1)or(img_binaire[i-1,j-1]==1)or(img_binaire[i-1,j+1]==1)or(img_binaire[i+1,j-1]==1)or(img_binaire[i+1,j+1]==1)):
                img_binaire[i,j] = 1
            else: 
                img_binaire[i,j] = 0

fig1 = plt.figure(1)
plt.subplot(1, 3, 1), plt.imshow(image_RGB), plt.title("Image initiale (3D)",  fontsize = 7)
plt.subplot(1, 3, 2), plt.imshow(image_gray), plt.title("Image (2D)",  fontsize = 7)
plt.subplot(1, 3, 3), plt.imshow(img_binaire), plt.title("Image binarisée",  fontsize = 7)
  
# # Opening pour enlever le bruit
# kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2)) 
# # kernel = np.ones((3,3), np.uint8)
# kernel_BH = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
# img_binaire = cv2.morphologyEx(img_binaire, cv2.MORPH_BLACKHAT, kernel_BH)
# img_binaire = cv2.morphologyEx(img_binaire, cv2.MORPH_CLOSE, kernel)

dst = cv2.cornerHarris(img_binaire,2,17,0.02)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
image_RGB[dst>0.01*dst.max()]=[0,0,255]

fig2 = plt.figure(2)
plt.subplot(1, 2, 1), plt.imshow(img_binaire), plt.title("Image binarisée après ouverture",  fontsize = 7)
# plt.subplot(1, 2, 2), plt.imshow(ouverture), plt.title("Image binarisée suite à opening",  fontsize = 7)
plt.subplot(1, 2, 2), plt.imshow(image_RGB), plt.title("Points detectés",  fontsize = 7)
  
