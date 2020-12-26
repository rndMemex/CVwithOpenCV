
# image_parser.py, usage load, crop and display an image with CV2 then save it into a file 
# Copyright() rndMemex@cantab.net 

import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



def saving_image(image, output):
    #save your image to output path using opencv
    path = output
    cv2.imwrite(os.path.join(path,'crop_image.jpg'), image)
    cv2.waitKey(0)

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help = 'Use -i flag with path to image as argument')
ap.add_argument('-s', '--save', required=True, help = 'Use -s flag with path to output as argument')
args = vars(ap.parse_args())

# Loading an image
image = cv2.imread(args["image"]) #B R G
cv2.imshow('Input', image)
cv2.waitKey(0) # to see a still image until you press something


# crop image
crop_image = image[100:1000, 10:500].copy()
cv2.imshow('Output', crop_image) # B R G input  
cv2.waitKey(0) # to see a still image until you press something
saving_image(crop_image, args["save"])