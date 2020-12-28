import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(image, cmap = None, fig_size = (10, 10)):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image, cmap = cmap)
    ax.axis('on')
    plt.show()


image = cv2.imread('../img/tree.jpg')
show_image(image)

# Performing Canny Edge Detection  on the original version of an image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)

# show our images
plt.figure(figsize=(20,10))
plt.subplot(221)
plt.imshow(gray, cmap ='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(222)
plt.imshow(eq, cmap ='gray')
plt.title('Image Equalisation'), plt.xticks([]), plt.yticks([])

# Canny Edge Detection for the Original
edges = cv2.Canny(gray, 10, 250)

plt.subplot(223),plt.imshow(edges,cmap = 'gray')
plt.title('Canny Edge of Original Image'), plt.xticks([]), plt.yticks([])


#Canny Edge Detection for the Equlized Image
edges_eq = cv2.Canny(eq, 10, 250)

plt.subplot(224),plt.imshow(edges_eq,cmap = 'gray')
plt.title('Canny Edge of Equalized Image'), plt.xticks([]), plt.yticks([])

plt.show()


