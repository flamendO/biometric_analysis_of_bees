import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
import skimage.io as sk
import skimage.color as skc
import skimage.filters as skf
import skimage.morphology as morpho
from skimage.util import img_as_float
import os

"""Rotation aile est un module permettant de retourner une image d'aile en fonction de
la plus grosse nervure de l'aile.

• Étapes : 

1- Seuillage pour ne récupérer que la grosse nervure
2- Binarisation
3- Récupération des lignes avec HoughLines
4- Calcul des angles par rapport à la verticale (on obtient une liste d'angles)
5- Prendre l'angle median 
6- Tourner l'image par rapport à cet angle

• Prototype :
rotate_wing : [image] -> [image_tourne, angle (degré)]
"""




def rotate_wing(img_path):
    
    ## ÉTAPE 1
    image = sk.imread(img_path)
    shape = image.shape
    
    
    if len(shape) == 2:  # On vérifie si l'image est en niveaux de gris
        image_gray = image
    else:
        image_gray = skc.rgb2gray(image)  
    
    A = np.zeros(shape)
    
    o = skf.threshold_otsu(image_gray) # Seuillage
    o = o/1.5
    image_gray[image_gray < o] = 0
    image_gray[image_gray > o] = 1

    
    rectan = morpho.rectangle(10, 3)
    image_ouv = morpho.binary_closing(image_gray, rectan)
    image_f = morpho.binary_opening(image_ouv, rectan)
    image_ero = morpho.binary_dilation(image_f, morpho.square(5))
    image_ero = morpho.binary_dilation(image_f, morpho.rectangle(13, 3))

    ## ÉTAPE 2
    seuil = 0.5  
    image_binaire = (image_ero > seuil).astype(np.uint8)

    ## ÉTAPE 3
    
    lines = cv2.HoughLinesP(image_binaire, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_binaire, (x1, y1), (x2, y2), (255, 255, 255), 1)

        ## ÉTAPE 4
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        
        ## ÉTAPE 5
        ang = np.abs(np.median(angles))
    

        ## ÉTAPE 6
        image_rot = rotate(image, -(90-ang), resize=False, center=None, order=None,
                           mode='constant', cval=1)

        return image_rot, ang
    else:
        return image, 0







## EXEMPLE (à décommenter)

# img_path = "./aile_test_rotate_6.png"
# image = sk.imread(img_path)
# image_redressee, angle_rotation = rotate_wing(image)


# plt.imshow(image_redressee)
# plt.title("Image redressée")
# plt.show()
