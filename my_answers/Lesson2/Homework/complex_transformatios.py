# TODO: Your code below
# create a pipeline where you apply different transformations

angles= [25, 45, 60, 120, 180, 360, -45, -90 ]
# transformations = ["rotate", "cropping", "scale", "translate", "mask", "brighten" ]
# "rotate", "cropping", "scale", "translate", "mask"
def pipeline_one(path):
    image = np.flip(cv2.imread(path), axis =2)
    # rotate
    print("ROTATING....")
    (height, width) = image.shape[:2]
    (cX, cY) = (width / 2, height / 2 )
    angle = random.choice(angles)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height))
    plt.subplot(1,2,2),plt.imshow(rotated)
    plt.title("Rotated by {} Degrees".format(angle)), plt.xticks([]), plt.yticks([])
    plt.show()

    # crop
    print("CROPPING...")
    cropped = rotated[85:250, 85:220] # h, w 
    plt.figure(figsize=(20,10))
    plt.subplot(2,2,2),plt.imshow(cropped)
    plt.title('Cropped'), plt.xticks([]), plt.yticks([])
    plt.show()

    # scale
    print("SCALING...")
    resized_width = 1280

    # calculating ratio of new image to old image
    ratio = resized_width / width
    dim = (resized_width, int(height * ratio))

    resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)

    print(f'Original shape: {cropped.shape} vs Resized shape {resized.shape}')
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1), plt.imshow(cropped) 
    plt.title('Original')
    plt.subplot(1,2,2), plt.imshow(resized)
    plt.title('Resized')
    plt.show()
    
    # translate
    print("TRANSLATING...")
    M = np.float32([[1, 0, 25], [0, 1, 50]])
    shifted = cv2.warpAffine(resized, M, (resized.shape[1], resized.shape[0]))

    plt.figure(figsize=(20,10))
    plt.subplot(1,2,2),plt.imshow(shifted)
    plt.title('Translated'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    # mask
    print("MASKING...")

    mask = np.zeros(shifted.shape[:2], dtype="uint8")
    cv2.circle(mask, (145, 200), 100, 255, -1)
    masked = cv2.bitwise_and(shifted, shifted, mask=mask)
    plt.subplot(1,2,2),plt.imshow(masked)
    plt.title("Masked"), plt.xticks([]), plt.yticks([])
    plt.show()
    
pathImage = '../img/beach.png'

pipeline_one(pathImage)


# Pipeline 2
# "crop", "rotate", "translate", "brighten",
    
def pipeline_two(path):
    image = np.flip(cv2.imread(path), axis =2)
    #crop
    print("CROPPING...")
    cropped = image[85:250, 85:220] # h, w 
    plt.figure(figsize=(20,10))
    plt.subplot(2,2,2),plt.imshow(cropped)
    plt.title('Cropped'), plt.xticks([]), plt.yticks([])
    plt.show()

    # rotate
    print("ROTATING....")
    (height, width) = cropped.shape[:2]
    (cX, cY) = (width / 2, height / 2 )
    angle = random.choice(angles)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(cropped, M, (width, height))
    plt.subplot(1,2,2),plt.imshow(rotated)
    plt.title("Rotated by {} Degrees".format(angle)), plt.xticks([]), plt.yticks([])
    plt.show()
    # translate
    print("TRANSLATING...")
    M = np.float32([[1, 0, 25], [0, 1, 50]])
    shifted = cv2.warpAffine(rotated, M, (rotated.shape[1], rotated.shape[0]))

    plt.figure(figsize=(20,10))
    plt.subplot(1,2,2),plt.imshow(shifted)
    plt.title('Translated'), plt.xticks([]), plt.yticks([])
    plt.show()
    # brighten
    print("BRIGHTEN...")
    M = np.ones(shifted.shape, dtype = "uint8") * 100
    added = cv2.add(shifted, M)
    plt.figure(figsize=(20,10))
    plt.subplot(2,2,1),plt.imshow(shifted)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(added)
    plt.title('Added'), plt.xticks([]), plt.yticks([])
    plt.show()



    
pathImage = '../img/beach.png'

pipeline_two(pathImage)
