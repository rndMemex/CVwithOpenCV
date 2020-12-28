import cv2
import numpy as np
import skimage
import matplotlib.pyplot as plt

def show_image(image, cmap = None, fig_size = (10, 10)):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image, cmap = cmap)
    ax.axis('off')
    plt.show()



image = cv2.imread('../img/license_plate.png')
show_image(image)

#Canny Edge Detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
wide = cv2.Canny(blurred, 10, 200)
mid = cv2.Canny(blurred, 30, 150)
tight = cv2.Canny(blurred, 240, 250)

# show the edge maps
plt.figure(figsize=(20,10))
plt.subplot(221),plt.imshow(blurred,cmap = 'gray')
plt.title('Blurred'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(wide,cmap = 'gray')
plt.title('Wide Edge Map'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(mid,cmap = 'gray')
plt.title('Mid Edge Map'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(tight,cmap = 'gray')
plt.title('Tight Edge Map'), plt.xticks([]), plt.yticks([])
plt.show()

plt.show()