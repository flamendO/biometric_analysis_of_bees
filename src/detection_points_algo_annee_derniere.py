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
import skimage.morphology as skm
import numpy as np
from math import sqrt
import skimage.util as sku
import skimage.color as skc
import skimage.feature as skf
import skimage.filters as skff

np.set_printoptions(threshold=sys.maxsize)


plt.rcParams['image.cmap'] = 'gray'


main_pathname = Path.cwd()


   



#traitement de l'image pour qu'elle soit binaire et qu'on récupère le skelette
def pre_traitement_image (img):
    seuil = 128
    img = ~ img 
    img = (img < seuil).astype(np.uint8)
    img = skm.skeletonize(img)
    img = ~ img 
    return (img) 



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
    
    
    # Marquer les points sur l'image
    for coordonnees in liste_coord:
        plt.scatter(coordonnees[1], coordonnees[0], color='red', marker='o')
    
    plt.show()
    
    return (liste_coord)



#fonction qui calcule l'indice cubital
def detection_automatique_cubital (img):
    
    print('Traitement pour indice cubital : ')
    img = detection_pattern_cubital(img)   #on travaille maintenant sur le pattern souhaité
    #img = pre_traitement_image(img)
    
    #premier appel à corner_harris
    a = 0.001
    b = 0.01
    
    list_coord = cornerharris(img, a, b)
    taille = len(list_coord)
    #print ("le nombre de points détectés au début est : ",taille)
    
    
    while ( taille != 3 ): 
        
        if ( taille == 0 ):
            print ("Pas assez de points détectés pour indice cubital")
            break 
  
        b = b + 0.005   
        list_coord = cornerharris(img,a,b)
        taille = len(list_coord)
        
    #print ("le nombre de points détectés à la fin est : ",taille)
    #print ("Meilleurs paramètres pour cette image :",a,b)
    
    #calcul de l'indice cubital 
       
    dist = []
    dist.append(np.sqrt(((list_coord[0][0] - list_coord[1][0])**2) + (list_coord[0][1] - list_coord[1][1])**2))
    dist.append(np.sqrt(((list_coord[0][0] - list_coord[2][0])**2) + (list_coord[0][1] - list_coord[2][1])**2))
    dist.append(np.sqrt(((list_coord[1][0] - list_coord[2][0])**2) + (list_coord[1][1] - list_coord[2][1])**2))
    print("Les distances pour l'indice cubital sont :", dist)
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
    elif (indice_cubital>1.55) & (indice_cubital<5) :
        print("L'indice cubital est compris entre 1,55 et 2, il faut réaliser la transgression discoïdale")

    return (dist)


#fonction qui calcule l'indice anthem
def detection_automatique_anthem (img, dist2):
    
    print('Traitement pour indice anthem : ')

    img = detection_pattern_anthem(img)   #on travaille maintenant sur le pattern souhaité
    #img = pre_traitement_image(img)
    
    #premier appel à corner_harris
    a = 0.001
    b = 0.01
    
    list_coord = cornerharris(img, a, b)
    taille = len(list_coord)
    #print ("le nombre de points détectés au début est : ",taille)
    
    
    while ( taille != 2 ): 
        
        if ( taille == 0 ):
            print ("Pas assez de points détectés pour indice cubital")
            break 
              
        b = b + 0.005     
        list_coord = cornerharris(img,a,b)
        taille = len(list_coord)
        
    #print ("le nombre de points détectés à la fin est : ",taille)
    #print ("Meilleurs paramètres pour cette image :",a,b)
    
    
    #calcul de l'indice anthem
    dist = []
    dist.append(np.sqrt(((list_coord[0][0] - list_coord[1][0])**2) + (list_coord[0][1] - list_coord[1][1])**2))
    #dist.append(np.sqrt(((list_coord[0][0] - list_coord[2][0])**2) + (list_coord[0][1] - list_coord[2][1])**2))
    #dist.append(np.sqrt(((list_coord[1][0] - list_coord[2][0])**2) + (list_coord[1][1] - list_coord[2][1])**2))
    print("Les distances pour l'indice anthem sont :", dist)
    A = max(dist)
    
    #on récupère la longeur utile qui a été trouvée dans l'indice cubital
    B = max(dist2)

    indice_anthem = A/B
    print("L'indice anthem est : ", indice_anthem)    
    
    




#fonction pour extraire le pattern cubital -> utilisation de la fonction matchTemplate
def detection_pattern_cubital(image): 
    
    #on prend un pattern de référence qu'on a extrait à la main sur une image 
    #puis on s'en sert pour chercher le même pattern dans l'image à traiter 
    pattern_ref = skio.imread(main_pathname / '../images /Masque_cubital.png')
    pattern_ref = pattern_ref[:, :, :3]
    pattern_ref = skc.rgb2gray(pattern_ref)

    
    
    result = skf.match_template(image, pattern_ref, mode = 'mean')
    
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    # Afficher l'image originale et la zone correspondant au motif
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
    
    ax1.imshow(pattern_ref, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('Template')
    
    ax2.imshow(image, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('Image')
    # Afficher un rectangle autour de la zone correspondante
    hcoin, wcoin = pattern_ref.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)
    
    ax3.imshow(result, cmap=plt.cm.gray)
    ax3.set_axis_off()
    ax3.set_title('Matched Result')
    # Marquer le maximum de corrélation
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
    
    plt.show()
    
    matched_area = image[y:y+hcoin, x:x+wcoin]

    # Afficher uniquement la zone de l'image correspondant au motif
    plt.imshow(matched_area, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Matched Area')
    plt.show()
 
    
    return (matched_area)



#fonction pour extraire le pattern anthem
def detection_pattern_anthem(image):
    
    pattern_ref = skio.imread(main_pathname / '../images /Masque_anthem.png')
    pattern_ref = pattern_ref[:, :, :3]
    pattern_ref = skc.rgb2gray(pattern_ref)

    
    
    result = skf.match_template(image, pattern_ref, mode = 'mean')
      
    
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    # Afficher l'image originale et la zone correspondant au motif
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
    
    ax1.imshow(pattern_ref, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('Template')
    
    ax2.imshow(image, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('Image')
    # Afficher un rectangle autour de la zone correspondante
    hcoin, wcoin = pattern_ref.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)
    
    ax3.imshow(result, cmap=plt.cm.gray)
    ax3.set_axis_off()
    ax3.set_title('Matched Result')
    # Marquer le maximum de corrélation
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
    
    plt.show()
    
    matched_area = image[y:y+hcoin, x:x+wcoin]

    # Afficher uniquement la zone de l'image correspondant au motif
    plt.imshow(matched_area, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Matched Area')
    plt.show()
    
    return (matched_area)




'''
Fonction permettant de transformer une image d'aile segmentée, éventuellement
après la rotation. Le principe se base sur une reconstruction géodésique.

On travaille sur une image HSV, car après tests, le channel S permet d'éliminer
certains bruits parasites sur l'image.

Le fonctionnement est le suivant : on applique un seuil sur l'image de départ,
ce qui nous permet d'éliminer les tâches claires. On fait une érosion pour
éliminer les petites tâches foncées. Le résultat obtenu est appelé mask.

On fabrique ensuite un masque (mask) en refaisant une érosion sur marker. Le
but est de dilater marker, puis de chercher les points communs entre mask et
marker. On reconstruit une partie des objets présents sur mask.

On peut répéter ces opérations pour reconstruire progressivement les nervures
de mproche en proche, sans reconstruire les défauts.

On applique ensuite la fonction squeletonize, pour récupérer des contours fins.
'''


def skelly(img):
    img_hsv = skc.rgb2hsv(img)
    img_sat = img_hsv[:, :, 1]
    img_centered = (img_sat - np.mean(img_sat)) / np.std(img_sat)

    # plot(img_centered)

    tresh = img_centered > 0.05*np.max(img_centered)
    tresh = tresh.astype(np.uint8)

    # plot(tresh)

    square5 = skm.square(10)
    square3 = skm.square(4)

    mask = tresh
    marker = skm.erosion(tresh, square5)
    geo = marker

    for i in range(1, 300):
        dil = skm.dilation(geo, square3)
        geo = np.minimum(dil, mask)
        # plot(geo) # pour afficher la reconstruction, décommenter cette ligne

    geo = skm.closing(geo, skm.square(5))
    skeleton = skm.thin(geo)*1  # thin ou bien skeletonize ?
    # plot(skeleton)
    return skeleton





'''
Boucle utile pour le test en masse, il faut se placer dans un dossier contenant
des images d'ailes segmentées, appelées ailes1.png, ailes2.png, ...
'''
'''
for i in range(0, 13):
    s = 'ailes{0}.png'.format(i)
    img_a = img_as_float(skio.imread(s))
    img = img_a[:, :, :3]
    plot(skelly(img))
'''




def detection_points (chemin):
    
    #ailes filtrées/Ruche_5_aile_4.jpg exemple de chemin
    plt.close('all')
    
    img = chemin
    img = img[:, :, :3]
    
    img = skelly(img)
    img = ~ img 
    
    #skio.imsave(main_pathname/'../images/img.png', img)
    
    
    plt.figure()
    plt.title('la')
    plt.imshow(img)
    
    dist = detection_automatique_cubital(img)
    detection_automatique_anthem(img, dist)
