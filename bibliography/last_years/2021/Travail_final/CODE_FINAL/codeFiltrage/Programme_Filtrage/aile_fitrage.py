import skimage.io as skio
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

#--- Fonctions ---#

def recherche_x_y(I,x_depart,y_depart,val_test,val_nouvelle):
    if I[x_depart,y_depart] == val_test:
        I[x_depart,y_depart] = val_nouvelle
        return(x_depart,y_depart)
    else:
        decalage_recherche = 1    
        while(True):
            if I[x_depart+decalage_recherche,y_depart] == val_test:
                I[x_depart+decalage_recherche,y_depart] = val_nouvelle
                return(x_depart+decalage_recherche,y_depart)
            elif I[x_depart-decalage_recherche,y_depart] == val_test:
                I[x_depart-decalage_recherche,y_depart] = val_nouvelle
                return(x_depart-decalage_recherche,y_depart)
            elif I[x_depart,y_depart+decalage_recherche] == val_test:
                I[x_depart,y_depart+decalage_recherche] = val_nouvelle
                return(x_depart,y_depart+decalage_recherche)
            elif I[x_depart,y_depart-decalage_recherche] == val_test:
                I[x_depart,y_depart-decalage_recherche] = val_nouvelle
                return(x_depart,y_depart-decalage_recherche)
            else:
                decalage_recherche += 1

def modifier_x_y_compteur(compteur):
    if compteur==0:
        return(round(l_max/2),round(c_max/2))
    elif compteur==1:
        return(round(l_max/2),round(c_max/2 + c_max/5))
    elif compteur==2:
        return(round(l_max/2 + l_max/5),round(c_max/2 + c_max/5))
    elif compteur==3:
        return(round(l_max/2 + l_max/5),round(c_max/2))
    elif compteur==4:
        return(round(l_max/2 + l_max/5),round(c_max/2 - c_max/5))
    elif compteur==5:
        return(round(l_max/2),round(c_max/2 - c_max/5))
    elif compteur==6:
        return(round(l_max/2 - l_max/5),round(c_max/2 - c_max/5))
    elif compteur==7:
        return(round(l_max/2 - l_max/5),round(c_max/2))
    elif compteur==8:
        return(round(l_max/2 - l_max/5),round(c_max/2 + c_max/5))
    else:
        return(round(l_max/2),round(c_max/2))

def nettoyage_en_croix(I,l,c,d,val_reference):
    for i in range(l-d):
        for j in range(c-d):
            val_gauche_haute = I[i,j]
            val_droite_haute = I[i+d,j]
            val_gauche_bas = I[i,j+d]
            val_droite_bas = I[i+d,j+d]
            if val_gauche_haute == val_reference:
                if val_gauche_haute == val_droite_haute:
                    for k in range(d-1):
                        I[i+k+1,j] = val_gauche_haute
                if val_gauche_haute == val_gauche_bas:
                    for k in range(d-1):                    
                        I[i,j+k+1] = val_gauche_haute
            if val_droite_bas == val_reference:
                if val_droite_haute == val_droite_bas:
                    for k in range(d-1):                    
                        I[i+d,j+k+1] = val_droite_haute
                if val_gauche_bas == val_droite_bas:
                    for k in range(d-1):                    
                        I[i+k+1,j+d] = val_gauche_bas

def coloriage_croix_liste(indice_x,indice_y,I,l,c,val_test,val_nouvelle):
    L = [(indice_x,indice_y)]
    while L != []:
        (x_actuel,y_actuel) = L[-1]
        L = L[:-1]
        if x_actuel<l-1:
            if I[x_actuel+1,y_actuel]==val_test:
                I[x_actuel+1,y_actuel]=val_nouvelle
                L.append((x_actuel+1,y_actuel))
        if x_actuel>0:
            if I[x_actuel-1,y_actuel]==val_test:
                I[x_actuel-1,y_actuel]=val_nouvelle
                L.append((x_actuel-1,y_actuel))
        if y_actuel<c-1:
            if I[x_actuel,y_actuel+1]==val_test:
                I[x_actuel,y_actuel+1]=val_nouvelle
                L.append((x_actuel,y_actuel+1))
        if y_actuel>0:
            if I[x_actuel,y_actuel-1]==val_test:
                I[x_actuel,y_actuel-1]=val_nouvelle
                L.append((x_actuel,y_actuel-1))

def coloriage_diag_liste(indice_x,indice_y,I,l,c,val_test,val_nouvelle):
    L = [(indice_x,indice_y)]
    while L != []:
        (x_actuel,y_actuel) = L[-1]
        L = L[:-1]
        if x_actuel<l-1:
            if I[x_actuel+1,y_actuel]==val_test:
                I[x_actuel+1,y_actuel]=val_nouvelle
                L.append((x_actuel+1,y_actuel))
        if x_actuel>0:
            if I[x_actuel-1,y_actuel]==val_test:
                I[x_actuel-1,y_actuel]=val_nouvelle
                L.append((x_actuel-1,y_actuel))
        if y_actuel<c-1:
            if I[x_actuel,y_actuel+1]==val_test:
                I[x_actuel,y_actuel+1]=val_nouvelle
                L.append((x_actuel,y_actuel+1))
        if y_actuel>0:
            if I[x_actuel,y_actuel-1]==val_test:
                I[x_actuel,y_actuel-1]=val_nouvelle
                L.append((x_actuel,y_actuel-1))
        if x_actuel<l-1 and y_actuel<c-1:
            if I[x_actuel+1,y_actuel+1]==val_test:
                I[x_actuel+1,y_actuel+1]=val_nouvelle
                L.append((x_actuel+1,y_actuel+1))
        if x_actuel>0 and y_actuel<c-1:
            if I[x_actuel-1,y_actuel+1]==val_test:
                I[x_actuel-1,y_actuel+1]=val_nouvelle
                L.append((x_actuel-1,y_actuel+1))
        if x_actuel<l-1 and y_actuel>0:
            if I[x_actuel+1,y_actuel-1]==val_test:
                I[x_actuel+1,y_actuel-1]=val_nouvelle
                L.append((x_actuel+1,y_actuel-1))
        if x_actuel>0 and y_actuel>0:
            if I[x_actuel-1,y_actuel-1]==val_test:
                I[x_actuel-1,y_actuel-1]=val_nouvelle
                L.append((x_actuel-1,y_actuel-1))

#--- Main ---#

t1 = time.time()

plt.close('all')

# Parameters
filename = "Ruche_21.jpg"
dirpath = r"images"
filepath = join(dirpath,filename)
decalage = 5
image_couleur = skio.imread(filepath)
x_debut = 0
x_fin = 600
y_debut = 1300
y_fin = 1558

# Choix de la taille de l'image
plt.figure(1)
plt.imshow(image_couleur)
image_couleur = image_couleur[x_debut:x_fin,y_debut:y_fin]

I1 = image_couleur[:,:,0]*0.298 + image_couleur[:,:,1]*0.586 + image_couleur[:,:,2]*0.114
I1 = np.array(I1,dtype=np.uint8)


I2 = cv2.adaptiveThreshold(I1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,25)

plt.figure(2)
plt.imshow(I2,'gray')

shape_image = I2.shape
l_max = shape_image[0]
c_max = shape_image[1]

I3 = np.array(I2,dtype=np.int16)
I4_reference = np.copy(I3)


# On veut nettoyer une cellule de l'aile ne y mettant que des pixels blancs sans les points noirs
compteur=0
not_succeed = True
while(not_succeed):
    print('Test actuel : '+str(compteur+1))
    (x,y)= modifier_x_y_compteur(compteur) # On choisit un point de départ en espérant
                                           # qu'il soit dans une cellule sinon on recommence
                                           # avec un autre point de départ
    
    (x,y) = recherche_x_y(I3,x,y,255,256)  # On cherche un point blanc de l'image
                                           # autour du point de départ et on change
                                           # sa valeur de 255 à 256 (pour l'identifier)

    
    coloriage_croix_liste(x,y,I3,l_max,c_max,255,256) # Si un point à 255 est collé 
                                                      # à un point vallant 256 il vaut
                                                      # alors 256
                                                      # Donc toute la cellule aura des
                                                      # points vallant soit 0 soit 256
    
    nettoyage_en_croix(I3,l_max,c_max,decalage,256) # si on a             256  ?   ?   ?  256
                                                    # alors on modifie en 256 256 256 256 256
    
    if np.sum(I3==256)<l_max*c_max/2 : # Permet de vérifier qu'on a colorié une cellule et
                                       # non le fond de l'image (perte d'information)
        not_succeed = False
        rate = False
        (x,y)= modifier_x_y_compteur(compteur)
        print('\nTest reussie : '+str(compteur+1))
    else:                              # Si on a nettoyé le fond on recommence avec un nouveau
                                       # Point de départ
        print('oui')
        compteur+=1
        I3 = np.copy(I4_reference)

    if compteur>8:                      # Si on a toujours pas trouvé de cellule alors on arrête
                                        # et la filtrage est raté
        not_succeed = False
        print("Toujours pas trouvé")
        rate = True
    


if rate == False:                   # Si on a réussi le filtrage, on va chercher un point noir
                                    # de l'aile
             
    for i in range(l_max):          # On remet les points qui valent 256 à 255
        for j in range(c_max):
            if I3[i,j]>255:
                I3[i,j]=255


    plt.figure(3)
    plt.imshow(I3,'gray')

    t2 = time.time()
    print('\nDurée première étape : '+str(round(t2-t1,2))+' s')

    I5 = np.copy(I3)

    (x,y) = recherche_x_y(I3,x,y,0,-1) # On recherche un point noir en partant de la cellule nettoyée
                                       # ce sera alors un point de l'aile et on le met à -1 
                                       # pour l'identifier
                                       
    coloriage_diag_liste(x,y,I5,l_max,c_max,0,-1) # Si un point à 0 est collé 
                                                  # à un point vallant -1 il vaut alors -1
                                                  # donc tous les points de l'aile sont à -1  

    for i in range(l_max):                      # Les points toujours à 0 sont donc du bruit
        for j in range(c_max):                  # on les mets à 255 (blanc)
            if I5[i,j] == 0:
                I5[i,j] = 255
            
    for i in range(l_max):                      # On mets les points de l'aile : -1 à 0
        for j in range(c_max):
            if I5[i,j] == -1:
                I5[i,j] = 0
            
    plt.figure(4)
    plt.imshow(I5,'gray')

    t3 = time.time()
    print('Durée seconde étape : '+str(round(t3-t2,2))+' s')
    