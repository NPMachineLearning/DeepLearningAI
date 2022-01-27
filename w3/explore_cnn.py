from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

# load the ascent image
ascent_image = misc.ascent()

# show image
plt.imshow(ascent_image)
plt.grid(False)
plt.axis(False)
plt.gray()
plt.show()

# copy image to numpy array
image_transformed = np.copy(ascent_image)

# get size of image
size_x = image_transformed.shape[0]
size_y = image_transformed.shape[1]

# create convolutional filter
# filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0] ]
# filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
filter = [ [-1, -1, 0], [-1, 0, 1], [0, 1, 1] ]
weight = 0.5

# apply convolution ignore 1 pixel
for x in range(1, size_x-1):
    for y in range(1, size_y-1):
        convolution = 0.0
        convolution = convolution + (ascent_image[x - 1, y - 1] * filter[0][0])
        convolution = convolution + (ascent_image[x - 1, y] * filter[0][1])
        convolution = convolution + (ascent_image[x - 1, y + 1] * filter[0][2])
        convolution = convolution + (ascent_image[x, y - 1] * filter[1][0])
        convolution = convolution + (ascent_image[x, y] * filter[1][1])
        convolution = convolution + (ascent_image[x, y + 1] * filter[1][2])
        convolution = convolution + (ascent_image[x + 1, y - 1] * filter[2][0])
        convolution = convolution + (ascent_image[x + 1, y] * filter[2][1])
        convolution = convolution + (ascent_image[x + 1, y + 1] * filter[2][2])

        # multiply weight
        convolution = convolution * weight

        # cap between 0 ~ 255
        if convolution<0:
            convolution = 0
        if convolution>255:
            convolution = 255

        image_transformed[x, y] = convolution

# show image
plt.gray()
plt.grid(False)
plt.axis(False)
plt.imshow(image_transformed)
plt.show()

# max pool
new_x = int(size_x/2)
new_y = int(size_y/2)

newImage = np.zeros((new_x, new_y))

for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = []
        pixels.append(image_transformed[x, y])
        pixels.append(image_transformed[x + 1, y])
        pixels.append(image_transformed[x, y + 1])
        pixels.append(image_transformed[x + 1, y + 1])

        newImage[int(x/2), int(y/2)] = max(pixels)

plt.gray()
plt.grid(False)
plt.axis(False)
plt.imshow(newImage)
plt.show()