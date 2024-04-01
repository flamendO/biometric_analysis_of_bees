# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:15:53 2024

@author: Johny
"""

import skimage as sk 
import matplotlib.pyplot as plt
import matplotlib
import skimage as ski
import numpy as np
from skimage import io 
from skimage import exposure
from skimage import color
from skimage.filters import try_all_threshold, threshold_otsu, median
from skimage import util
from skimage import transform
from skimage import segmentation
from skimage import morphology , util
from skimage.filters import rank
from skimage import filters
from random import randint
import cv2



def plot(img):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    
def func(I):
    I  = I[:,:,:3]
    I = color.rgb2gray(I)
    
    # Thresholding
    
    I_hyst = filters.apply_hysteresis_threshold(I, low = 0.60, high =0.9 )
    # Filtre pour la détection de nervures
    
    I_meij = filters.meijering(I)
    I_sato = filters.sato(I)
    I_meij*=255
    I_meij = I_meij.astype(np.uint8)
    I_treat = pre_traitement_image(I_meij)
    
    # Affichage des images 
    
    plot(I)
    plot(I_meij)
    plot(I_sato)
    plot(I_hyst)
    plot(I_treat)
    
    return I 


    
def pre_traitement_image (img):
    seuil = 170
    img = ~ img  # Inversion des couleurs
    img = (img < seuil).astype(np.uint8) # Binarisation de l'image
    img = morphology.skeletonize(img) # Réduction de la largeur de nervures

    return (img) 


# Boucle pour afficher le résultat des traitements

for k in range(97,111):
    func(io.imread("./ailes_segmentees/"+chr(k)+".png"))
    
    
    