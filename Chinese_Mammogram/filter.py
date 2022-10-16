import numpy as np
import glob
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters
from tensorflow.python.keras.preprocessing.image import img_to_array





image = skimage.io.imread("E:/ET-A/SDP/FINAL/Chinese_Mammogram/Malignant/5.jpeg")

fig, ax = plt.subplots()
plt.imshow(image)
plt.show()

# convert the image to grayscale
gray_image = skimage.color.rgb2gray(image)

# blur the image to denoise
blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
fig, ax = plt.subplots()
plt.imshow(blurred_image, cmap='gray')
plt.show()

# show the histogram of the blurred image
# histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))
# plt.plot(bin_edges[0:-1], histogram)
# plt.title("Graylevel histogram")
# plt.xlabel("gray value")
# plt.ylabel("pixel count")
# plt.xlim(0, 1.0)
# plt.show()
t = skimage.filters.threshold_otsu(blurred_image) 
#print(“Found automatic threshold t = {}.”).format(t)) ~~~ {: .language-python}

# create a binary mask with the threshold found by Otsu's method
binary_mask = blurred_image > t

fig, ax = plt.subplots()
plt.imshow(binary_mask, cmap='gray')
plt.show()
plt.savefig("Threshold_segmentation")
i=img_to_array(binary_mask)
#i=np.asarray(i)
print(i)
