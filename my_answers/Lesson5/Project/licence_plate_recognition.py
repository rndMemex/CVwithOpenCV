import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pytesseract

def show_image(image, cmap = None, fig_size = (10, 10)):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image, cmap = cmap)
    ax.axis('on')
    plt.show()




image = cv2.imread('../img/licence_plate_raw.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(20,10))
plt.subplot(2,2,1),plt.imshow(gray, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

kernel_size= (11,11)

# loop over the kernels and apply a "closing" operation to the image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
white_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

mask = np.zeros(white_hat.shape[:2], dtype="uint8")
cv2.rectangle(mask, (280, 230), (480, 290), 255, -1)
masked = cv2.bitwise_and(white_hat, white_hat, mask=mask)
plt.figure(figsize=(20,10))
plt.subplot(221),plt.imshow(masked,cmap = 'gray')

# plt.subplot(2,2,i+2),plt.imshow(white_hat, cmap = 'gray')
plt.title(f"Black Hat: ({kernel_size[0]}, {kernel_size[1]})"), plt.xticks([]), plt.yticks([])
plt.show()
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = masked[topx:bottomx+1, topy:bottomy+1]
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
plt.figure(figsize=(20,10))
plt.subplot(221),plt.imshow(Cropped, cmap = 'gray')
plt.show()
text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("License plate Number:",text)
img = cv2.resize(image,(500,300))
Cropped = cv2.resize(Cropped,(400,200))
