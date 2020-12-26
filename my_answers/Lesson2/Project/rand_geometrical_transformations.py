# random_geometrical_tranformations.py
# Description: Applying multiple tranformations at once on one image
# Usage: python rand_geometriccal_tranformations.py
# Copyright() rndMemex@cantab.net

import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random 


def saving_image(image, output, image_name):
    #save your image to output path using opencv
    path = output
    im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(path, image_name), im_bgr)
    cv2.waitKey(0)

# function block
def crop(image, output):
    #crop
    print("CROPPING...")
    cropped = image[85:250, 85:220] # h, w 
   # cropped = np.flip(cropped, axis =2)
    plt.figure(figsize=(20,10))
    plt.subplot(2,2,2),plt.imshow(cropped)
    plt.title('Cropped'), plt.xticks([]), plt.yticks([])
    plt.show()

    saving_image(cropped, output, 'cropped_image.jpg')

def rotate(image, output):
    # rotate stuff 
    (height, width) = image.shape[:2]
    (cX, cY) = (width / 2, height / 2 )
    angle = random.choice(angles)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height))
    # rotated = np.flip(rotated, axis = 2)
    plt.subplot(1,2,2),plt.imshow(rotated)
    plt.title("Rotated by {} Degrees".format(angle)), plt.xticks([]), plt.yticks([])
    plt.show()
    saving_image(rotated, output, 'rotated_image.jpg')


def bright(image, output):
    # add stuff
    bright_num = [50, 90, 100, 150, 200]
    rand_brightnes = random.choice(bright_num)
    print("BRIGHTEN by {}...".format(rand_brightnes))
    M = np.ones(image.shape, dtype = "uint8") * rand_brightnes
    added = cv2.add(image, M)
    # added= np.flip(added, axis = 2)
    plt.figure(figsize=(20,10))
    plt.subplot(2,2,1),plt.imshow(image)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(added)
    plt.title('Added'), plt.xticks([]), plt.yticks([])
    plt.show()
    saving_image(added, output, 'brghtened_image.jpg')

def substract(image, output):
    #  substract stuff
    subtract_num = [15, 50, 25, 55, 75, 29]
    rand_darkness = random.choice(subtract_num)
    print("DARKENED by {}...".format(rand_darkness))
    M = np.ones(image.shape, dtype = "uint8") * rand_darkness
    subtracted = cv2.subtract(image, M)
    # subtracted = np.flip(subtracted, axis =2)
    plt.figure(figsize=(20,10))
    plt.subplot(2,2,2),plt.imshow(subtracted)
    plt.title('Subtracted'), plt.xticks([]), plt.yticks([])
    plt.show()
    saving_image(subtracted, output, 'darkened_image.jpg')


def translate(image, output):
    # translate stuff
    # translate
    print("TRANSLATING...")
    M = np.float32([[1, 0, 25], [0, 1, 50]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # shifted = np.flip(shifted, axis =2)
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,2),plt.imshow(shifted)
    plt.title('Translated'), plt.xticks([]), plt.yticks([])
    plt.show()
    saving_image(shifted, output, 'translated_image.jpg')

def scale(image, output):
    # scale stuff
    print("SCALING...")
    resized_width = [1280, 1920, 400, 960]
    rand_width = random.choice(resized_width)

    # calculating ratio of new image to old image
    ratio = rand_width / width
    dim = (rand_width, int(height * ratio))

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    print(f'Original shape: {image.shape} vs Resized shape {resized.shape}')
    #  resized = np.flip(resized, axis =2)
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1), plt.imshow(image) 
    plt.title('Original')
    plt.subplot(1,2,2), plt.imshow(resized)
    plt.title('Resized')
    plt.show()
    saving_image(resized, output, 'scaled_image.jpg')

def flipOne(image, output):
    # flip stuff
    print("FLIPPING HORIZONTALLY...")
    # flip the image horizontally
    hflipped = cv2.flip(image, 1)
    #  hflipped = np.flip(hflipped, axis=2)

    plt.figure(figsize=(20,10))
    plt.subplot(2,2,2),plt.imshow(hflipped)
    plt.title('Flipped Horizontally'), plt.xticks([]), plt.yticks([])
    plt.show()

    saving_image(hflipped, output, 'flippedX_image.jpg')

def flipTwo(image, output):
    print("FLIPPING VERTICALLY...")
    # flip the image vertically
    vflipped = cv2.flip(image, 0) 
    #  vflipped = np.flip(vflipped, axis = 2)  
    plt.figure(figsize=(20,10))
    plt.subplot(2,2,3),plt.imshow(vflipped)
    plt.title('Flipped Vertically'), plt.xticks([]), plt.yticks([])
    plt.show()
    saving_image(vflipped, output, 'flippedY_image.jpg')   

def flipThree(image, output):
    print("FLIPPING HORIZONTALLY & VERTICALLY...")
    # flip the image along both axes
    hvflipped = cv2.flip(image, -1)
    # hvflipped = np.flip(hvflipped, axis =2)
    plt.figure(figsize=(20,10))
    plt.subplot(2,2,4),plt.imshow(hvflipped)
    plt.title('Flipped Horizontally & Vertically'), plt.xticks([]), plt.yticks([])
    plt.show()
    saving_image(hvflipped, output, 'flippedXY_image.jpg')

def mask(image, output):
    # mask stuff
    print("MASKING...")

    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (145, 200), 100, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    # masked = np.flip(masked, axis = 2)
    plt.subplot(1,2,2),plt.imshow(masked)
    plt.title("Masked"), plt.xticks([]), plt.yticks([])
    plt.show()

    saving_image(masked, output, 'masked_image.jpg')



ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help = 'Use -i flag with path to image as argument')
ap.add_argument('-s', '--save', required=True, help = 'Use -s flag with path to output as argument')
args = vars(ap.parse_args())

# Loading an image
image = cv2.imread(args["image"]) #B R G
output = args["save"]
cv2.imshow('Input Image', image)
cv2.waitKey(0)
image = np.flip(image, axis =2)
angles= [25, 45, 60, 120, 180, 360, -45, -90 ]
(height, width) = image.shape[:2]


transformationsSwitcher = {
    1: crop, 
    2: rotate, 
    3: bright, 
    4: substract, 
    5: translate, 
    6: scale, 
    7: flipOne,
    8: flipTwo,
    9: flipThree, 
    10: mask
}
samples = random.sample(range(1, 11),10)
print(samples)
# invoke switch block 
for i in samples:
    print("{}th Image".format(1))
    print("random transformation: {}".format(i))
    transformationsSwitcher[i](image, output)








