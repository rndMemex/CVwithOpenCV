import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(image, cmap = None, fig_size = (10, 10)):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image, cmap = cmap)
    ax.axis('off')
    plt.show()

#Black Hat
image = cv2.imread('../img/licence_plate_raw.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(20,10))
plt.subplot(2,2,1),plt.imshow(gray, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

kernel_sizes = [(5, 5), (8, 8), (11, 11)]

print("BLACK HAT USING cv2.MORPH_RECT")
# loop over the kernels and apply a "black hat" operation to the image
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    white_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    plt.subplot(2,2,i+2),plt.imshow(white_hat, cmap = 'gray')
    plt.title(f"Black Hat: ({kernel_size[0]}, {kernel_size[1]})"), plt.xticks([]), plt.yticks([])

plt.show()

plt.figure(figsize=(20,10))
plt.subplot(2,2,1),plt.imshow(gray, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

print("BLACK HAT USING cv2.MORPH_ELLIPSE")
# loop over the kernels and apply a "black hat" operation to the image
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    white_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    plt.subplot(2,2,i+2),plt.imshow(white_hat, cmap = 'gray')
    plt.title(f"Black Hat: ({kernel_size[0]}, {kernel_size[1]})"), plt.xticks([]), plt.yticks([])

plt.show()

plt.figure(figsize=(20,10))
plt.subplot(2,2,1),plt.imshow(gray, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
print("BLACK HAT USING cv2.MORPH_CROSS")
# loop over the kernels and apply a "black hat" operation to the image
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    white_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    plt.subplot(2,2,i+2),plt.imshow(white_hat, cmap = 'gray')
    plt.title(f"Black Hat: ({kernel_size[0]}, {kernel_size[1]})"), plt.xticks([]), plt.yticks([])

plt.show()


