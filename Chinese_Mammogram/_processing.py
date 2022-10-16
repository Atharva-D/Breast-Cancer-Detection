import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf


DIRECTORY = r"E:/ET-A/SDP/Chinese_Mammogram"
CATEGORIES = ["Benign", "Malignant"]

data = []
labels = []
image_path=[]

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image_path.append(img_path)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)



print(len(data))
# print(image_path)

# pre_processed_data=[]

# for i in range(len(data)):
#     image = skimage.io.imread(data[i])

#     # fig, ax = plt.subplots()
#     # plt.imshow(image)
#     # plt.show()

#     # convert the image to grayscale
#     gray_image = skimage.color.rgb2gray(image)

#     # blur the image to denoise
#     blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
#     # fig, ax = plt.subplots()
#     # plt.imshow(blurred_image, cmap='gray')
#     # plt.show()

#     # show the histogram of the blurred image
#     #histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))
#     # plt.plot(bin_edges[0:-1], histogram)
#     # plt.title("Graylevel histogram")
#     # plt.xlabel("gray value")
#     # plt.ylabel("pixel count")
#     # plt.xlim(0, 1.0)
#     # plt.show()
    
#     #t = skimage.filters.threshold_otsu(blurred_image) 
#     #binary_mask = blurred_image > t
#     #b_mask = img_to_array(binary_mask)
#     #image = preprocess_input(b_mask)
#     im = np.asarray(blurred_image)
#     img = preprocess_input(im)
#     pre_processed_data.append(blurred_image)
#     #print(“Found automatic threshold t = {}.”).format(t)) ~~~ {: .language-python}

#     # create a binary mask with the threshold found by Otsu's method
    

#     # fig, ax = plt.subplots()
#     # plt.imshow(binary_mask, cmap='gray')
#     # plt.show()
#     # plt.savefig("Threshold_segmentation")

# print(len(pre_processed_data))

# print(type(data))
# print(data)
# print(data[2])

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, img_as_ubyte
from skimage.filters import threshold_multiotsu

preprocessed_data=[]
for i in image_path:
    image = io.imread(i)
    threshold=threshold_multiotsu(image,classes=4)
    regions=np.digitize(image,bins=threshold)
    output=img_as_ubyte(regions)
    #plt.imsave("Plots/Otsu_segmented.jpeg",output)
    #plt.imshow(image)
    #plt.plot(image)
    preprocessed_data.append(output)


print(len(preprocessed_data))


#print(preprocessed_data)
plt.imsave("Plots/Otsu_segmented_new.jpeg",preprocessed_data[1])
plt.imshow(preprocessed_data[1])
plt.plot(preprocessed_data[1])
plt.show()


