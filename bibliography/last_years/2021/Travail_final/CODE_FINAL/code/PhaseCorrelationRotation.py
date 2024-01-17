# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 18:22:19 2021

@author: leontine
"""


from matplotlib import pyplot as plt
from sklearn import cluster
import skimage.io as skio
from os.path import join
from skimage.morphology import binary_dilation, binary_erosion
import skimage as sk
from skimage.filters import gabor, gaussian
import math as m
import numpy as np
import scipy.ndimage as scnd
import cv2
import time
from PIL import Image
import os
from scipy import fftpack
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale, resize
from skimage.color import rgb2gray
from skimage.filters import window, difference_of_gaussians
from scipy.fftpack import fft2, fftshift
from skimage.util import img_as_float

plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

"""
=========================== fonctions utiles pour l'algorithme principal ===========================
"""

def decoupe(img, h_p, l_p, h_f, l_f): # permet de découper des images gray 
    crop_img = np.copy(img[ h_p:h_f, l_p:l_f ])
    return crop_img

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma= 2, verbose=False): # construction d'une gaussienne en 2D
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    if verbose:
        plt.imshow(kernel_2D, interpolation='none',cmap='gray')
        plt.title("Image")
        plt.show()
    return kernel_2D

def distance(e1,e2) : # permet de calculer la distance euclidienne entre deux vecteurs
    dist = 0
    for i in range (len(e1)) :
        dist = dist + (e1[i]-e2[i])**2
    return (m.sqrt(dist))

def sans_rotation( rts_image, image ) : # permet de placer "droit" une aile mal orientée qui generait la detection de motif
    originale = rts_image.copy()
    # rts_image = resize(rts_image, image.shape)
    rts_image = cv2.resize(rts_image, image.shape[::-1], interpolation=cv2.INTER_AREA)

    
    # First, band-pass filter both images
    image = difference_of_gaussians(image, 5, 20)
    rts_image = difference_of_gaussians(rts_image, 5, 20)
       
    # window images
    wimage = image * window('hann', image.shape)
    rts_wimage = rts_image * window('hann', image.shape)
    
    # work with shifted FFT magnitudes
    image_fs = np.abs(fftshift(fft2(wimage)))
    rts_fs = np.abs(fftshift(fft2(rts_wimage)))
    
    # Create log-polar transformed FFT mag images and register
    shape = image_fs.shape
    radius = shape[0] // 8  # only take lower frequencies
    warped_image_fs = warp_polar(image_fs, radius=radius, output_shape=shape, scaling='log', order=0)
    warped_rts_fs = warp_polar(rts_fs, radius=radius, output_shape=shape, scaling='log', order=0)
    
    warped_image_fs = warped_image_fs[:shape[0] // 2, :]  # only use half of FFT
    warped_rts_fs = warped_rts_fs[:shape[0] // 2, :]
    shifts, error, phasediff = phase_cross_correlation(warped_image_fs, warped_rts_fs, upsample_factor=10)
    
    # Use translation parameters to calculate rotation and scaling parameters
    shiftr, shiftc = shifts[:2]
    recovered_angle = (360 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    shift_scale = np.exp(shiftc / klog)
    rotated = rotate(originale, - shiftr, resize = False )
    print(recovered_angle)
    return shiftr, rotated

"""
=========================== Algorithme principal ===============================================
objectif :
    Pour calculer l'indice cubital, on a besoin d'extraire une zone bien précise de l'aile : 
    on veut donc identifier ce motif d'interet ("token"), présent dans l'aile puis l'en extraire. 
"""

nombre_ailes = 1
num_ruche = 38 # numéro de la ruche 
nb_token =  20 # nombre de token de la base de données 
num_ruche_ref = 59 # test pour la rotation
num_aile_ref = 13 # test pour la rotation

shape = np.zeros((2,nb_token))

filename_ref = r'Ruche_'+str(num_ruche_ref)+'_aile_'+str(num_aile_ref)+'.jpg'
dirpath_ref = r'..\code\photos\Ailes\Ruche_'+str(num_ruche_ref)+'_filter'
filepath_ref = join(dirpath_ref, filename_ref)
image_ref = skio.imread(filepath_ref, as_gray = True )

for num in range( 1, nombre_ailes+1 ) : 
    num_aile_traitee = 17 # numéro de l'aile (cf. base de donnée) dont on veut extraire le motif
    
    filename = r'Ruche_'+str(num_ruche)+'_aile_'+str(num_aile_traitee)+'.jpg'
    dirpath = r'..\code\photos\Ailes\Ruche_'+str(num_ruche)+'_filter'
    filepath = join(dirpath, filename)
    img_rgb = cv2.imread(filepath)
    img_gray0 = skio.imread(filepath, as_gray = True )
    img_gray = sans_rotation( img_gray0, image_ref )[1]
    img_gray2 = img_gray.copy()
    img_gray3 = img_gray.copy()
    
  
    fig6 = plt.figure(6)
    plt.subplot(1, 2, 1),  plt.imshow(img_gray0), plt.title("originale"),
    plt.subplot(1, 2, 2),  plt.imshow(img_gray), plt.title("retournée"),

    
    taille_img = img_gray.shape
    points = np.zeros((2,nb_token)) # tableau qui, pour chaque token identifié dans l'image, enregistre (x,y) du centre du token
    img_montagne = np.zeros( taille_img )
    
    TF_aile = fftpack.fft2( img_gray, shape = taille_img ) # utile pour la corrélation de phase

    for k in range(1, nb_token+1): # on parcourt tous les tokens de la base de donnés
        
        # base de token particuliere
        # filename_token = r'Ruche_'+str(num_base_token)+'_token_'+str(k)+'.jpg' 
        # dirpath_token = r'..\code\photos\token\filter\indice_cubital\indice_cubital_total\Ruche_'+str(num_base_token)
        
        #  base complete de tokens
        filename_token = str(k)+'.jpg' 
        dirpath_token = r'..\code\photos\token\filter\indice_cubital\indice_cubital_total\Base'
        filepath_token = join(dirpath_token, filename_token)    
        template = skio.imread(filepath_token, as_gray = True) # template : c'est le token n°i et il sert de référence dans la recherche de motif
        
        w, h = template.shape[::-1]
        shape[0, k-1] = h
        shape[1, k-1] = w
        img = img_gray.copy()
        
        # correlation de phase 
        TF_template_conj = np.conj(fftpack.fft2( template, shape = taille_img ))
        difference_de_phase = ( TF_template_conj*TF_aile )/abs( TF_template_conj*TF_aile )
        correlation_de_phase = abs( fftpack.ifft2( difference_de_phase ))
        
        top_left = np.unravel_index( correlation_de_phase.argmax(), correlation_de_phase.shape ) # coin supérieur gauche
        bottom_right = (top_left[1] + w, top_left[0] + h)
        # cv2.rectangle( img_rgb, top_left[::-1], bottom_right, (100, 0, 200), 1)
        cv2.rectangle( img_gray, top_left[::-1], bottom_right, 0 , 1)
        
        fig1 = plt.figure(1)
        plt.subplot(121),plt.imshow(correlation_de_phase)
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img_gray) # cmap = 'gray'
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle("Ensemble des motifs reconnus pour l'aile n°" + str(num_aile_traitee) + " de la ruche n°" + str(num_ruche), fontsize = 7 )
    
        # mettre le centre des motifs dans un tableau pour la répartition statistique
        x_center = bottom_right[1] - w//2
        y_center = bottom_right[0] - h//2
        points[0][k-1] = x_center
        points[1][k-1] = y_center
        
        # création de l'image "montagne" = répartition statistique
        if (min(template.shape)%2!=0) :
             kernel = gaussian_kernel(min(template.shape)+1)
        else :
            kernel = gaussian_kernel(min(template.shape))
        taille_kernel = min(kernel.shape)  
        for i in range (-int(taille_kernel/2),int(taille_kernel/2)) :
            for j in range (-int(taille_kernel/2),int(taille_kernel/2)) :
                if (x_center+i)<img_montagne.shape[0] :
                    if (y_center+j)<img_montagne.shape[1] :
                        img_montagne[x_center+i][y_center+j] += kernel[i+int(taille_kernel/2)][j+int(taille_kernel/2)]
        
         # création de l'image "montagne" = répartition statistique
        """
        kernel = gaussian_kernel(20)
        taille_kernel = 20
        for i in range (-int(taille_kernel/2),int(taille_kernel/2)) :
            for j in range (-int(taille_kernel/2),int(taille_kernel/2)) :
                if (x_center+i)<img_montagne.shape[0] :
                    if (y_center+j)<img_montagne.shape[1] :
                        img_montagne[x_center+i][y_center+j] += kernel[i+int(taille_kernel/2)][j+int(taille_kernel/2)]
        """
    fig2 = plt.figure(2)
    plt.subplot(1, 2, 1), plt.imshow( img_gray ), plt.title("Image contenant le token",  fontsize = 7)
    plt.subplot(1, 2, 2), plt.imshow( img_montagne ), plt.title("Probabilité de présence du token", fontsize = 7)
    
    #détection de la partie la plus claire de l'image :
    center_final_token = [ np.unravel_index(np.argmax(img_montagne, axis=None), img_montagne.shape)[0], np.unravel_index(np.argmax(img_montagne, axis=None), img_montagne.shape)[1] ]
    
    # on reforme un cadre pour le motif le plus ressemblant trouvé dans l'aile (pour récuperer les dimensions)
    indice = 0
    dist_0 = 10e6
    for l in range (0, nb_token) :
        dist = distance(center_final_token, points[:,l])
        if (dist < dist_0) :
            indice = l + 1 #indice nous donne le token à prendre dans l'image de base
            dist_0 = dist
    
    filename_token_final = str(indice)+'.jpg' 
    dirpath_token_final = r'..\code\photos\token\filter\indice_cubital\indice_cubital_total\Base'
    filepath_token_final = join(dirpath_token_final, filename_token_final)
    template_final = skio.imread(filepath_token_final, as_gray = True )
    
    # On refait le même traitement que precedemment mais uniquement pour le token optimal identifié dans l'aile
    shape_final = ( int(np.mean(shape[0,:])) + 5, int(np.mean(shape[1,:])) + 5)
    w2, h2 = shape_final[::-1]
    img_rgb2 = cv2.imread( filepath )
    img2 = skio.imread( filepath, as_gray = True )
    
    # correlation de phase 
    TF_template_final_conj = np.conj(fftpack.fft2( template_final, shape = taille_img ))
    difference_de_phase2 = ( TF_template_final_conj*TF_aile )/abs( TF_template_final_conj *TF_aile )
    correlation_de_phase2 = abs( fftpack.ifft2( difference_de_phase2 ))
    
    top_left2 = np.unravel_index( correlation_de_phase2.argmax(), correlation_de_phase2.shape ) # coin supérieur gauche
    bottom_right2 = (top_left2[1] + w2, top_left2[0] + h2)
    # cv2.rectangle( img_rgb2, top_left2[::-1], bottom_right2, (100, 0, 200), 1)
    cv2.rectangle( img_gray2, top_left2[::-1], bottom_right2, 0 , 1)
    
    fig3 = plt.figure(3)
    plt.subplot(121),plt.imshow(correlation_de_phase2) # cmap = 'gray'
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_gray2) # cmap = 'gray'
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle("motif reconnu pour l'aile n°" + str(num_aile_traitee) + " de la ruche n°" + str(num_ruche), fontsize = 7 )

    
    fig4 = plt.figure(4)
    plt.subplot(1, 2, 1), plt.imshow(img_gray2), plt.title("image avec le token optimal",  fontsize = 7)
    plt.subplot(1, 2, 2), plt.imshow(template_final), plt.title("token optimal identifié",  fontsize = 7)
    
    # Isolation de la zone selectionnée (token optimal)
    
    h_p = top_left2[1]
    h_f = top_left2[1] + w2
    l_p = top_left2[0]
    l_f = top_left2[0] + h2
    token_optimal = decoupe(img_gray3, l_p , h_p , l_f , h_f )*255
    
    fig5 = plt.figure(5)
    plt.subplot(1, 2, 1), plt.imshow(img_gray2), plt.title("image avec le token optimal",  fontsize = 7)
    plt.subplot(1, 2, 2), plt.imshow(token_optimal), plt.title("token extrait de l'aile traitée",  fontsize = 7)
    
    #Enregistrement dans une nouvelle image du token optimal 
    
    cv2.imwrite( r'..\code\photos\token_trouve\Ruche_'+str(num_ruche)+'\Ruche_'+str(num_ruche)+'_aile_'+str(num_aile_traitee)+'.jpg',token_optimal)
 