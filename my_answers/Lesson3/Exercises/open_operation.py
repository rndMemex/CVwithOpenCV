
def show_image(image, cmap = None, fig_size = (10, 10)):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image, cmap = cmap)
    ax.axis('off')
    plt.show()
# opening using different structuring elements

image = cv2.imread('../img/snowy_i.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(20,10))
plt.subplot(2,2,1),plt.imshow(gray, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

kernel_sizes = [(1, 1), (3, 3), (5, 5)]
 
# loop over the kernels and apply an "opening" operation to the image
print("OPENING USING cv2.MORPH_RECT")
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    plt.subplot(2,2,i+2),plt.imshow(opening, cmap = 'gray')
    plt.title(f"Opening: ({kernel_size[0]}, {kernel_size[1]})"), plt.xticks([]), plt.yticks([])

plt.show()

plt.figure(figsize=(20,10))
plt.subplot(2,2,1),plt.imshow(gray, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
print("OPENING USING cv2.MORPH_ELLIPSE")
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    plt.subplot(2,2,i+2),plt.imshow(opening, cmap = 'gray')
    plt.title(f"Opening: ({kernel_size[0]}, {kernel_size[1]})"), plt.xticks([]), plt.yticks([])

plt.show()

plt.figure(figsize=(20,10))
plt.subplot(2,2,1),plt.imshow(gray, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
print("OPENING USING cv2.MORPH_CROSS")
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    plt.subplot(2,2,i+2),plt.imshow(opening, cmap = 'gray')
    plt.title(f"Opening: ({kernel_size[0]}, {kernel_size[1]})"), plt.xticks([]), plt.yticks([])

plt.show()

#Answer: using morph_ellipse seems better for opening operation because the output
# is clearer
