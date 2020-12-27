# TODO: Load the seal image and calculate the gradient orientation and magnitude of the image at pixel 100, 100
img = cv2.imread('../img/seal.png')
show_image(img)
# solve exercise
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# compute gradients along the X and Y axis, respectively
sobelx = cv2.Sobel(gray, ddepth = cv2.CV_64F, dx=1, dy=0)
sobely = cv2.Sobel(gray, ddepth = cv2.CV_64F, dx=0, dy=1)

# the gx and gy images are now of the floating point data type
# so we need to take care to convert them back to an unsigned 8-bit
# integer representation so other opencv functions can utilise them
edge_gradient = np.sqrt((sobelx**2)+(sobely**2)) #magnitude
angle = np.arctan2(sobelx, sobely) * (180 / np.pi) % 180
print(edge_gradient[100, 100])
print(angle[100, 100])