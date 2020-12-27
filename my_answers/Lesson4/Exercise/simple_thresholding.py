# TODO: Apply the above three thresholding methods to another image. Which has performed better? - You may need to fine tune Hyperparameters

image = cv2.imread('../img/opencv_logo.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0) 

show_image(blurred, cmap='gray')


# Simple Thresholding 

# black rather than white.
(T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
# Threshold Binary
show_image(thresh, cmap = 'gray')

# finally, we can visualize only the masked regions in the image
bitwise_output = cv2.bitwise_and(image, image, mask=threshInv)
show_image(np.flip(image, axis = 2))
show_image(np.flip(bitwise_output, axis = 2))

# Otsu's Method

# apply Otsu's automatic thresholding -- Otsu's method automatically
# determines the best threshold value `T` for us
(T, threshInv) = cv2.threshold(blurred, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
print("Otsu's thresholding value: {}".format(T))
show_image(threshInv, cmap = 'gray') # same as cv2.imshow("Threshold", threshInv)
 
# finally, we can visualize only the masked regions in the image
bitwise_and = cv2.bitwise_and(image, image, mask=threshInv)
show_image(bitwise_and, cmap = 'gray') # same as cv2.imshow("Output", bitwise_and)


# Adaptive Thresholding
# instead of manually specifying the threshold value, we can use adaptive
# thresholding to examine neighborhoods of pixels and adaptively threshold
# each neighborhood -- in this example, we'll calculate the mean value
# of the neighborhood area of 25 pixels and threshold based on that value;
# finally, our constant C is subtracted from the mean calculation (in this
# case 15)
thresh = cv2.adaptiveThreshold(blurred, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 5)
show_image(thresh, cmap = 'gray') # same as cv2.imshow("OpenCV Mean Thresh", thresh)
 
# personally, I prefer the scikit-image adaptive thresholding, it just
# feels a lot more "Pythonic"
T = threshold_local(blurred, 33, offset=4, method="gaussian")
thresh = (blurred < T).astype("uint8") * 255
show_image(thresh, cmap = 'gray') # cv2.imshow("scikit-image Mean Thresh", thresh)