
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:40:32 2024

@author: boute
"""


from pathlib import Path
import skimage.io as skio
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np
from math import sqrt
import skimage.util as sku
import skimage.color as skc
import skimage.feature as skf
import skimage.filters as skff
import skimage.exposure as ske 
import skimage.restoration as skr 
import skimage.morphology as skm
import skimage.util as sku
import os

np.set_printoptions(threshold=sys.maxsize)
global indice_images
indice_images = 0

plt.rcParams['image.cmap'] = 'gray'


main_pathname = Path.cwd()




#fonction qui détecte les points aux intersection 
def cornerharris (image, para1, threshold):
    
    '''
    Fonction qui permet de detecter les intersections à l'aide de la fonction cornerHarris de OpenCV.
    
    L'image en entrée est en binaire (0, 1), codé en float32. Elle est pré-filtré avant son entrée dans cornerHarris par la fonction skelly (nettoyage et affinage). 
    On obtient en sortie la liste des coordonnées des intersections de l'image.
    
    Première étape = détéction des intersections grâce à cornerHarris (paramètres : image / taille du voisinage / 
                    paramètre d'ouverture du filtre de Sobel / paramètre libre de l'équation de détéction de Harris. 
                    Cette étape renvoie une image dont chaque pixel a une valeur élevée si forte probabilité que ce soit une intersection, 
                    faible valeur si faible probabilité d'intersection.
    
    Deuxième étape = seuillage de cette nouvelle image pour ne garder que les points qui sont des intersections.
    
    Troisième étape = transformation des groupes de points en un point singulier en chaque intersection.
    
    Quatrième étape = récupération des coordonnées des points d'intersection.
    '''
    
    image = np.float32(image)
    #image = pre_traitement_image(image)
    
    
    # Première étape
    points = cv2.cornerHarris(image, 5, 3, para1)  # Paramètres à régler
    
    # Deuxième étape
    points_thresh = (points > threshold) * 1 # Paramètres à régler
    points_thresh = sku.img_as_ubyte(points_thresh)
    
    # Troisième étape
    contours, _ = cv2.findContours(
        points_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Boucle à travers les tâches blanches autour des intersections pour trouver le centre
    for i, c in enumerate(contours):
    
        # Obtention des frontières des tâches blanches
        x, y, w, h = cv2.boundingRect(c)
        # Calcul des centres des tâches blanches
        cx = int(x + 0.5 * w)
        cy = int(y + 0.5 * h)
    
        # Dessin d'un pixel au milieu détecté
        fillPosition = (cx, cy)
        fillColor = (0, 0, 0)
        cv2.floodFill(points_thresh, None, fillPosition, fillColor,
                      loDiff=(10, 10, 10), upDiff=(10, 10, 10))
        points_thresh[cy, cx] = 255
    
    # Quatrième étape
    liste_coord = []
    shape = np.shape(points_thresh)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if points_thresh[i, j] == 255:
                liste_coord = liste_coord + [(i, j)]
                
    
    plt.figure()
    plt.imshow(image)
    global indice_images
    
    
    # Marquer les points sur l'image
    for coordonnees in liste_coord:
        plt.scatter(coordonnees[1], coordonnees[0], color='red', marker='o')
    plt.savefig('../tmp/'+str(indice_images)+'.png')
    indice_images = indice_images + 1
    plt.close()
    
    #plt.show()
    
    return (liste_coord)



#fonction qui calcule l'indice cubital
def detection_automatique_cubital (img):
    
    img = detection_pattern_cubital(img)   #on travaille maintenant sur le pattern souhaité
    img = skm.skeletonize(1-img)
    
    #premier appel à corner_harris
    a = 0.05
    b = 0.01
    
    list_coord = cornerharris(img, a, b)
    taille = len(list_coord)
    
    
    while ( taille != 3 ): 
        
        if ( taille == 0 or taille == 1 or taille == 2 ):
            return (0,0) 
  
        b = b + 0.005   
        list_coord = cornerharris(img,a,b)
        taille = len(list_coord)
        
    
    #calcul de l'indice cubital 
       
    dist = []
    dist.append(np.sqrt(((list_coord[0][0] - list_coord[1][0])**2) + (list_coord[0][1] - list_coord[1][1])**2))
    dist.append(np.sqrt(((list_coord[0][0] - list_coord[2][0])**2) + (list_coord[0][1] - list_coord[2][1])**2))
    dist.append(np.sqrt(((list_coord[1][0] - list_coord[2][0])**2) + (list_coord[1][1] - list_coord[2][1])**2))
    B = min(dist)
    for a in dist :
        if a != max(dist) :
            if a != min(dist) :
                A = a
    
    indice_cubital= A/B
    #Affichage de l'indice cubital pour l'aile sélectionnée

    return (dist, indice_cubital)


#fonction qui calcule l'indice anthem
def detection_automatique_anthem (img, dist2):
    

    if (dist2 == 0): #si la distance pour l'indice cubitale est nul, alors on a pas réussi à exploiter l'image donc on renvoit 0 aussi ici
        return 0

    img = detection_pattern_anthem(img)   #on travaille maintenant sur le pattern souhaité
    #img = pre_traitement_image(img)
    img = skm.skeletonize(1-img)
    #premier appel à corner_harris
    a = 0.05
    b = 0.01
    
    list_coord = cornerharris(img, a, b)
    taille = len(list_coord)
    #print ("le nombre de points détectés au début est : ",taille)
    
    
    while ( taille != 2 ): 
        
        if ( taille == 0 or taille == 1 ):
            return 0 # On retourne 0 si il y a une erreur
             
              
        b = b + 0.005     
        list_coord = cornerharris(img,a,b)
        taille = len(list_coord)
    
    
    #calcul de l'indice anthem
    dist = []
    dist.append(np.sqrt(((list_coord[0][0] - list_coord[1][0])**2) + (list_coord[0][1] - list_coord[1][1])**2))
    A = max(dist)
    
    #on récupère la longeur utile qui a été trouvée dans l'indice cubital
    B = max(dist2)

    indice_anthem = A/B    
    
    return indice_anthem
    




#fonction pour extraire le pattern cubital -> utilisation de la fonction matchTemplate
def detection_pattern_cubital(image): 
    
    matchs = []
    for k in range (1, 6) :
        pattern_ref = skio.imread(main_pathname / './images /Masque_cubital_{}.png'.format(k))
        pattern_ref = pattern_ref[:, :, :3]
        pattern_ref = skc.rgb2gray(pattern_ref)
        result = skf.match_template(image, pattern_ref, mode = 'mean')
        matchs.append(result)
    
    #on cherche le maximum des matchs pour les différents template
    max_unique_value = float('-inf')
    maxi = None
    
    for i in matchs :
        unique_values = set()
        for row in i:
            unique_values.update(row)
        current_max_unique_value = max(unique_values)
        
        if current_max_unique_value > max_unique_value:
            max_unique_value = current_max_unique_value
            maxi = i
    
    ij = np.unravel_index(np.argmax(maxi), maxi.shape)
    x, y = ij[::-1]

    # Afficher l'image originale et la zone correspondant au motif
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
    
    hcoin, wcoin = pattern_ref.shape
    
    matched_area = image[y:y+hcoin, x:x+wcoin]

    
 
    
    return (matched_area)



#fonction pour extraire le pattern anthem
def detection_pattern_anthem(image):
    
    matchs = []
    for k in range (1, 6) :
        pattern_ref = skio.imread(main_pathname / './images /Masque_anthem_{}.png'.format(k))
        pattern_ref = pattern_ref[:, :, :3]
        pattern_ref = skc.rgb2gray(pattern_ref)
        result = skf.match_template(image, pattern_ref, mode = 'mean')
        matchs.append(result)
    
    #on cherche le maximum des matchs pour les différents template
    max_unique_value = float('-inf')
    maxi = None
    
    for i in matchs :
        unique_values = set()
        for row in i:
            unique_values.update(row)
        current_max_unique_value = max(unique_values)
        
        if current_max_unique_value > max_unique_value:
            max_unique_value = current_max_unique_value
            maxi = i
      
    
    ij = np.unravel_index(np.argmax(maxi), maxi.shape)
    x, y = ij[::-1]


    hcoin, wcoin = pattern_ref.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    
    matched_area = image[y:y+hcoin, x:x+wcoin]


    
    return (matched_area)





#ajout de la fonction d'Alexis
def filtrage(img):
    
    img = skc.rgb2gray(img)
    img = ske.equalize_adapthist(img,clip_limit=0.01)
    img = skr.denoise_bilateral(img)
    img = skm.opening(img,footprint = skm.diamond(1))
    img = skm.black_tophat(img, footprint = skm.diamond(9))
    
    img = sku.img_as_uint(img)
    img = sku.img_as_ubyte(img)
    img = img/255
    # if(np.min(img)<0): #
    #     img = img - np.min(img) #
    #     img = img * int(255/np.max(img)) #

    # img = img.astype(np.uint8) #
    
    # img = cv2.Canny(img, 100,200) #
    # img = skm.binary_closing(img, footprint= skm.disk(3)) #
    seuil = 0.2
    img = (img < seuil)  #image binaire
    # plt.figure()
    # plt.imshow(img, cmap='gray')
    return img




def detection_point (chemin):
    
    
    plt.close('all')
    
    img = chemin
    img = img[:, :, :3]
    
    
    img = filtrage(img)
    
    #skio.imsave(main_pathname/'../images/img.png', img) # à dégager après
    
    
    dist,indice_cubital = detection_automatique_cubital(img)
    indice_anthem = detection_automatique_anthem(img, dist)

    return (indice_cubital, indice_anthem)
