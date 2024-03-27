import os
from wing_extraction import wing_extraction
from rotation_aile import rotate_wing
from detection_points_algo_annee_derniere import detection_points
import numpy as np
import matplotlib.pyplot as plt




# Chemin du fichier 
path  = '../images/1.jpg'

# Séparation des ailes 

indice = wing_extraction(path)

# Rotation des ailes 

images_list = [] # Liste avec les images detectées et orientées
angles = np.zeros((indice)) # Array avec les angles de rotation de chaque image si nécéssaire

for i in range(1,indice+1):
    image_tmp, angle_tmp = rotate_wing(str(os.getcwd()) + '/' + str(i) + '.png')
    angles[i-1] = angle_tmp
    images_list.append(np.array(image_tmp))

# suite 

for i in range (len(images_list)):
    plt.figure()
    plt.imshow(images_list[i])
    
    
#detection_points(images_list[4])





