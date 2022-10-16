import os
import io
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import UnidentifiedImageError
import glob
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters
from tensorflow.python.keras.preprocessing.image import img_to_array

DIRECTORY = r"E:/ET-A/SDP/Chinese_Mammogram"
CATEGORIES = ["Benign", "Malignant"]

data = []
labels = []
image_path=[]
#Pre_Processed_data=[]

try:
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image_path.append(img_path)
            image = io.imread(img_path)
            gray_image = skimage.color.rgb2gray(image)
            blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
            #t = skimage.filters.threshold_otsu(blurred_image) 
            #binary_mask = blurred_image > t
            im=img_to_array(blurred_image)
            #image = img_to_array(image)
            im = preprocess_input(im)
            data.append(im)
            labels.append(category)

except(UnidentifiedImageError,IOError, SyntaxError):
    print("All good")





print(len(data))
image = io.imread(image_path[1277])
plt.imshow(image)
plt.plot(image)
plt.show()

plt.imshow(data[1277])
plt.plot(data[1277])
plt.show()
# try:
#     for i in image_path:


#         image = skimage.io.imread(i)

#         # fig, ax = plt.subplots()
#         # plt.imshow(image)
#         # plt.show()

#         # convert the image to grayscale
#         gray_image = skimage.color.rgb2gray(image)

#         # blur the image to denoise
#         blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
#         #fig, ax = plt.subplots()
#         #plt.imshow(blurred_image, cmap='gray')
#         #plt.show()

#         # show the histogram of the blurred image
#         #histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))
#         #plt.plot(bin_edges[0:-1], histogram)
#         #plt.title("Graylevel histogram")
#         # plt.xlabel("gray value")
#         # plt.ylabel("pixel count")
#         # plt.xlim(0, 1.0)
#         # plt.show()
#         t = skimage.filters.threshold_otsu(blurred_image) 
#         #print(“Found automatic threshold t = {}.”).format(t)) ~~~ {: .language-python}

#         # create a binary mask with the threshold found by Otsu's method
#         binary_mask = blurred_image > t
#         im=img_to_array(binary_mask)
#         Pre_Processed_data.append(i)

#         # fig, ax = plt.subplots()
#         # plt.imshow(binary_mask, cmap='gray')
#         # plt.show()
#         # plt.savefig("Threshold_segmentation")

# except(IOError, SyntaxError) as e:
#     print("Bad file")



# print(len(Pre_Processed_data))