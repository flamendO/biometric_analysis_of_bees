# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:20:38 2021

@author: alice
"""

#Import des bibliothèques et fonctions utilisées
from matplotlib import pyplot as plt
from sklearn import cluster
import skimage.io as skio
from os.path import join
from skimage.morphology import binary_dilation, binary_erosion, skeletonize, opening
import skimage as sk
from skimage.filters import gabor, gaussian
import math as m
import numpy as np
import scipy.ndimage as scnd
import cv2
import time
from PIL import Image
import os
import numpy


plt.rcParams['image.cmap'] = 'gray'
plt.close('all')


#%%
#calcul du crossing number pour un point donné :
    # somme des 8 pixels environnants au pixel i,j de img

def crossing_number(img, i, j) :
    c_n = img[i-1][j-1] + img[i-1][j] + img[i-1][j+1] + img[i][j-1] + img[i][j+1] + img[i+1][j-1] + img[i+1][j] + img[i+1][j+1]
    return c_n ;

    #Dans notre cas d'utilisation, les pixels seront égaux à 1 ou 0 sur une image binarisée,
    #donc le crossing number sera le nombre de pixels environnants noirs.

#%%
#fonction qui calcule la moyenne de M, une liste de couple de coordonnées
def moyenne(M) :
    x = 0
    y = 0
    for i in range(len(M)) :
        x = x + M[i][0]
        y = y + M[i][1]
    return (x/len(M), y/len(M))

#calcul de la distance euclidienne entre e1 et e2
def distance(e1, e2):
    dist = 0
    for i in range(len(e1)):
        dist = dist + (e1[i]-e2[i])**2
    return(m.sqrt(dist))

#%%

#Choix des numéros de l'aile et de la ruche :
num_aile_traitee = 8
num_ruche = 18

#Importation du découpage de l'image correspondante afin de traiter l'indice cubital seulement :
filename = r'indice_cub_' + str(num_aile_traitee) + '_ruche_' + str(num_ruche) + '.png'
dirpath = r'..\code_final\'
filepath = join(dirpath, filename)
#filepath = 'C:/Users/alice/Documents/2A/Projet Abeilles/code/indice_cub_' + str(num_aile_traitee) + '_ruche_' + str(num_ruche) + '.png'

img_rgb = cv2.imread(filepath)
img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
n, m = img.shape

###Binarisation de l'image :
img_bin = img.copy()
#Choix d'un seuil
threshold = 200

for i in range (n) :
    for j in range (m) :
        if img[i][j] > threshold :
            img_bin[i][j] = 0
        else :
            img_bin[i][j] = 1


###Opérations morphologiques :
    #on extrait un élément structurant nommé kernel, à partir duquel nous réalisons une fermeture avant de récupérer le "squelette" de l'image.
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
img_closed = sk.morphology.closing(img_bin,kernel)
img_thin = sk.morphology.skeletonize(img_closed)

#Affichage
fig1 = plt.figure()
plt.subplot(1,4,1), plt.imshow(img), plt.title('Initiale')
plt.subplot(1,4,2), plt.imshow(img_bin), plt.title('Binarisée')
plt.subplot(1,4,3), plt.imshow(img_closed), plt.title('Fermeture')
plt.subplot(1,4,4), plt.imshow(img_thin), plt.title('Skeleton')


#Avant de travailler sur l'image obtenue, on remet l'image squelette sur une binarisation en {0,1} :
img_thin_bin = np.zeros((n,m))

for i in range (n) :
    for j in range (m) :
        if img_thin[i][j] == True :
            img_thin_bin[i][j] = 1

#Et on supprime son cadre qui résulte du découpage initial, et qui amène des perturbations
img_thin_bin[0][:] = 0
img_thin_bin[:,0] = 0
img_thin_bin[n-1][:] = 0
img_thin_bin[:,m-1] = 0

#Affichage de l'image finale sur laquelle on travaille ensuite :
fig2 = plt.figure()
plt.imshow(img_thin_bin), plt.title('Thin bin')

#%%
#On cherche ensuite à trouver les bifurcations/intersections pour localiser les points d'intérêt.
intersections = []

for i in range (5, n-5) :
    for j in range (5, m-5) :
        if (crossing_number(img_thin_bin, i, j) == 3) & (img_thin_bin[i,j] == 1) :
            intersections.append([i,j])
#Un pixel est ajouté dans intersections si il est lui même égal à 1, et si son crossing number vaut 3.


# Relevé des coordonnées des points d'intérêt en affinant la sélection :
    #on ajoute d'office les premier et dernier pixels de la liste intersections dans une nouvelle liste m,
    #divisée en 3 sous groupes correspondant aux 3 points à relever
m = [ [intersections[0]], [], [intersections[-1]] ]
k = 0
for i in range( 1, len(intersections)-1 ) : # on parcourt les couples de la liste
    if (np.linalg.norm(np.array(intersections[i]) - np.array(intersections[i+1]) ) < 9) :
        #on compare la distance entre deux pixels successifs de la liste intersections
         if (intersections[i] not in m[k]) :
             m[k].append(intersections[i])
         if (intersections[i+1] not in m[k]) :
             m[k].append(intersections[i+1])
    elif k < 2 :
        k = k+1


#Moyennage des groupes de m pour avoir un couple de coordonnées (x,y) par point d'intérêt :
moy = []
for i in range(len(m)) :
    if not m[i] :
        print("La détection a échoué")
    else :
        moy.append(moyenne(m[i]))


#%%
### Calcul de l'indice cubital d'après les distances entre les points déterminés par leurs 2 coordonnées :
if (len(moy) == 3) :
    a = np.array(moy[0])
    b = np.array(moy[1])
    c = np.array(moy[2])
    dist = []
    dist.append(np.linalg.norm(a - b))
    dist.append(np.linalg.norm(a - c))
    dist.append(np.linalg.norm(b - c))
    B = min(dist)
    for a in dist :
        if a != max(dist) :
            if a != min(dist) :
                A = a

    indice_cubital= A/B
    #Affichage de l'indice cubital pour l'aile sélectionnée
    print("indice cubital :", indice_cubital )
    if (indice_cubital<1.55) :
        print(" Il s'agit d'une abeille noire")
    elif (indice_cubital>2):
        print("Il ne s'agit pas d'une abeille noire")
    elif (indice_cubital>2) & (indice_cubital<5) :
        print("L'indice cubital est compris entre 1,55 et 2, il faut réaliser la transgression discoïdale")
else :
    print("Erreur etape finale")
