import os
import numpy as np
import skimage as sk
import skimage.io as skio
import cv2
from skimage.draw import line
import skimage.filters as skf
import skimage.color as skc
import matplotlib.pyplot as plt
from time import time
import skimage.morphology as morpho
from skimage.transform import hough_line, hough_line_peaks, rotate
import shutil

"""
La fonction wing_extraction prends en paramètre une image avec les ailes d'abeilles et renvoie le nombre d'image d'aile
d'abeilles extraites, elle enregistre également les images extraites sous le nom "idx.png" avec idx un entier allant de
1 au Nombre total d'images détectées.
"""

def wing_extraction(filename): # Avec l'extension
    
    nom_fichier = filename 

    image = skio.imread(nom_fichier)
    size = image[:,:,0].shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pour obtenir les meilleurs résultats sur le seuillage on commence par flouter l'image pour diminuer le bruit autrement,
    # le seuillage considérerait les moindres points foncés issus du bruit sur les zones claires et inversement. C'est 
    # le bruit sel & poivre que l'on veut réduire.


    image = skio.imread(nom_fichier)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    med = cv2.medianBlur(image,15)
    haus = cv2.bilateralFilter(med, 20, 50, 50) # Blurry the image 
    
    # Le seuillage donnant les meilleurs résultats est le seuillage d'Otsu, on l'effectue alors.

    seu, th = cv2.threshold(haus,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    # Ensuite, on veut uniquement les rectangles donc on réalise un opening pour supprimer les artefacts sur l'image.
    # On le fait deux fois, pour supprimer le plus d'artefacts possible.

    th2 = th.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17,17))
    opening = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

    opening2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) # Il est écrit "close" car cela dépend si l'image est 
    # blanche sur fond noir ou noir sur fond blanche mais l'opération est la même. 

    # L'opération d'opening réduit l'épaisseur des bords des rectangles donc on va les élargir avec une 


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    final2 = cv2.erode(opening2,kernel,iterations = 4)
    final3 = cv2.dilate(final2,kernel,iterations = 4)
    
    # A partir de la on va commencer la détection des rectangles, on ouvre l'image initiale pour ensuite 
    # découper les rectangles par dessus.
    image = skio.imread(nom_fichier)

    # A l'aide d'openCV on détecte automatiquement les positions des contours. contours est une liste
    contours =cv2.findContours(final3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

    # Ensuite on boucle sur les contours puis on stock ceux étant des rectangles dans la liste cntrRect.
    cntrRect = []
    idx = 0
    print(os.getcwd())
    # verification pour le stockage des fichiers
    if os.path.isdir("./save"):
        shutil.rmtree("./save")
        print("Fichier supprimé")

    os.mkdir("./save")
    os.chdir("./save")

    for i in contours:
            epsilon = 0.05*cv2.arcLength(i,True) # Cette fonction calcul le périmètre du contour i, puisque l'on veut 
            # un contour fermé, on ajoute le paramètre True.
            # Then this perimeter is used to calculate the epsilon value for cv2.approxPolyDP() function with a 
            # precision factor for approximating the rectangle.
        
            approx = cv2.approxPolyDP(i,epsilon,True) # Le périmetre précédemment calculé permet alors d'estimer
            # le rectangle le plus proche de ce dernier
            if len(approx) == 4: # on vérifie que le polygone à 4 cotés 
                cv2.drawContours(image,cntrRect,-1,(0,255,0),2)
                cntrRect.append(approx)
                x, y, w, h = cv2.boundingRect(i)
                if w > 400 or h > 600 or w < 100 or h < 100 : # On connait les tailles approximatives de chaque rectangle
                    # donc on supprime ceux trop éloignés. 
                    continue
                roi = image[y:y + h, x:x + w]
                idx += 1
                cv2.imwrite(str(os.getcwd()) + '/' + str(idx) + '.png', roi)
                
    os.chdir("../")            
    return idx


# Fonction pour extraire les chemins de tous les fichiers dans un répértoire avec leur chemin complet.
def extract_file_paths(directory):
    # Liste les fichiers dans le répertoire
    files = os.listdir(directory)
    
    # Extrait le chemin entier
    file_paths = [os.path.join(directory, file) for file in files if file != '.DS_Store' and os.path.isfile(os.path.join(directory, file))]
    
    return file_paths



