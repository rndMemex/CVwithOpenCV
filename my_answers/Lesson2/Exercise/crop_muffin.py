# crop_muffin.py, 
# Crop out a muffin from a picture
# Copyright() rndmemex@cantab.net


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def display_canvas(image, cmap = None, fig_size = (10,10)):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image, cmap = cmap)
    ax.axis('on')
    plt.show()

def saving_image(image, output):
    #save your image to output path using opencv
    path = output
    cv2.imwrite(os.path.join(path,'crop_out.jpg'), image)
    cv2.waitKey(0)


image = np.flip(cv2.imread('../img/dog_muffin.jpg'), axis=2)
# display_canvas(image)

mask = np.zeros(image.shape[:2], dtype='uint8')
cv2.rectangle(mask, (550,195), (620, 265), 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask)
display_canvas(masked)
masked = np.flip(masked, axis =2)
saving_image(masked, '../img')

