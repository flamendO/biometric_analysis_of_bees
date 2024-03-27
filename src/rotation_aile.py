import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
import skimage.io as sk
import skimage.color as skc
import skimage.filters as skf
import skimage.morphology as morpho
from skimage.util import img_as_float

def rotate_wing(img_path):
    image = sk.imread(img_path)
    shape = image.shape
    
    # Vérifier si l'image est en niveaux de gris
    if len(shape) == 2:  
        image_gray = image
    else:
        image_gray = skc.rgb2gray(image)  
    
    A = np.zeros(shape)
    # Seuillage
    o = skf.threshold_otsu(image_gray)
    o = o/1.5
    image_gray[image_gray < o] = 0
    image_gray[image_gray > o] = 1

    
    rectan = morpho.rectangle(10, 3)
    image_ouv = morpho.binary_closing(image_gray, rectan)
    image_f = morpho.binary_opening(image_ouv, rectan)
    image_ero = morpho.binary_dilation(image_f, morpho.square(5))
    image_ero = morpho.binary_dilation(image_f, morpho.rectangle(10, 3))

    seuil = 0.5  
    image_binaire = (image_ero > seuil).astype(np.uint8)
    
    # plt.imshow(image_binaire, cmap='gray')
    
    # plt.show()
    
    lines = cv2.HoughLinesP(image_binaire, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        max_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_binaire, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # plt.imshow(image_binaire, cmap='gray')
        # plt.show()
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        
        ang = np.abs(np.median(angles))
    

        
        image_rot = rotate(image, -(90-ang), resize=False, center=None, order=None,
                           mode='constant', cval=1)

        return image_rot, ang
    else:
        return image, 0

#EXEMPLE :
# img_path = "./aile_test_rotate_6.png"
# image = sk.imread(img_path)
# image_redressee, angle_rotation = rotate_wing(image)


# plt.imshow(image_redressee)
# plt.title("Image redressée")
# plt.show()
