from wing_extraction import wing_extraction
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image  
import PIL 
import skimage.io as skio


wing_extraction('./images/1.jpg')

image = skio.imread('./save/14.png')

print(image.shape)

