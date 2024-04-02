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


def wing_extraction(filename): # Avec l'extension
    
    nom_fichier = filename 

    # plt.figure(figsize=(15,15))
    image = skio.imread(nom_fichier)
    # mask = skio.imread('./mask.jpg')
    size = image[:,:,0].shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image.shape

    # To get the best result with the Threshold we first want to blurry the image to highlights the countours, otherwise,
    # the threshold will consider too much areas


    # fig = plt.figure(figsize=(12,12))
    # plt.subplot(2,2,1)
    image = skio.imread(nom_fichier)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    med = cv2.medianBlur(image,15) # Reduce noise (salt  & pepper )
    #plt.imshow(med, cmap='gray')
    #plt.xticks([])
    #plt.yticks([])

    haus = cv2.bilateralFilter(med, 20, 50, 50) # Blurry the image 
    

    seu, th = cv2.threshold(haus,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    

    # Result without the threshold
    # seu_, th_ = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    

    # Then we want to keep the rectangles only so we do morpholigcal operation : opening 
    th2 = th.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17,17))
    opening = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17,17))
    opening2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # To ensure that the rectangles are correctly detected we widen their countours

    #fig = plt.figure(figsize=(15,15))
    #plt.subplot(1,2,1)
    final = opening2.copy()


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    final2 = cv2.erode(final,kernel,iterations = 4)


    final3 = cv2.dilate(final2,kernel,iterations = 4)

    
    cont1 = final3.copy()

    image = skio.imread(nom_fichier)
    # cnts = cv2.findContours(cont1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours =cv2.findContours(cont1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

    #Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
    cntrRect = []
    idx = 0
    if os.path.isdir("./save"):
        file_paths = [f"./save/{file}" for file in os.listdir("./save")]
        for file_path in file_paths:
            os.remove(file_path)
    else:
        os.mkdir("./save")

    os.chdir("./save")

    for i in contours:
            epsilon = 0.05*cv2.arcLength(i,True) # Calculate the perimeter of the contour, second parameter is True 
            # because we want the countour to be closed
            # Then this perimeter is used to calculate the epsilon value for cv2.approxPolyDP() function with a 
            # precision factor for approximating the rectangle.
            approx = cv2.approxPolyDP(i,epsilon,True)
            if len(approx) == 4:
                
                cv2.drawContours(image,cntrRect,-1,(0,255,0),2)
                cntrRect.append(approx)
                x, y, w, h = cv2.boundingRect(i)
                if w > 400 or h > 600 or w < 100 or h < 100 :
                    continue
                roi = image[y:y + h, x:x + w]
                idx += 1
                cv2.imwrite(str(os.getcwd()) + '/' + str(idx) + '.png', roi)
                
    
                   
    # print('Number of rectangles detected : '+ str(idx - 1))
    # cv2.imwrite('test_im.png', image)
    # contours = skio.imread('./test_im.png')
    return idx

def extract_file_paths(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Extract full file paths, excluding .DS_Store files
    file_paths = [os.path.join(directory, file) for file in files if file != '.DS_Store' and os.path.isfile(os.path.join(directory, file))]
    
    return file_paths


# directory = './photo_bleu/'
# file_paths = extract_file_paths(directory)
# for file_path in file_paths:
#    wing_extraction(file_path)
#    print(file_path)



