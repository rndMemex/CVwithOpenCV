import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(image, cmap = None, fig_size = (10, 10)):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image, cmap = cmap)
    ax.axis('on')
    plt.show()

def plot_histogram(image, title, mask=None):
    # grab the image channels, initialize the tuple of colors and
    # the figure
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

image = np.flip(cv2.imread('../img/licence_plate_raw.png'), axis=2)
show_image(image)


# Plotting a histogram for an image
plt.figure(figsize=(20,10))


plt.subplot(221)
plt.imshow(np.flip(image, axis = 2))
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(222)
plot_histogram(image, "Histogram for Original Image")

# Plotting a histogram for the mask of an image
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (280, 230), (480, 290), 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask)
plt.figure(figsize=(20,10))


plt.subplot(223)
plt.imshow(np.flip(masked, axis = 2))
plt.title('Masked'), plt.xticks([]), plt.yticks([])



plt.subplot(224)
# compute a histogram for our image, but we'll only include pixels in
# the masked region
plot_histogram(image, "Histogram for Masked Image", mask=mask)
plt.show()
