# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.image as img
# import glob
# import io
# from PIL import Image
# import cv2

# Benign_Images=glob.glob("E:/ET-A/SDP/Chinese_Mammogram/Benign/*.jpeg")
# Benign_Images_1=glob.glob("E:/ET-A/SDP/Chinese_Mammogram/Benign/*.jpg")
# #print(len(Benign_Images_1))
# #print(len(Benign_Images))
# Benign_Images.extend(Benign_Images_1)
# print(len(Benign_Images))

# Malignant_Images=glob.glob("E:/ET-A/SDP/Chinese_Mammogram/Malignant/*.jpeg")
# print(len(Malignant_Images))

# image=img.imread(Malignant_Images[0])
# image=image.resize(256,256)
# plt.plot(image)

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"E:/ET-A/SDP/Chinese_Mammogram"
CATEGORIES = ["Benign", "Malignant"]

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)



print(len(data))
print(len(labels))

datagen = ImageDataGenerator(validation_split=0.20, height_shift_range=0.10,width_shift_range=0.10,rotation_range=30,rescale=1/255.)








img=load_img('E:/ET-A/SDP/Chinese_Mammogram/Benign/1.jpeg')

X=img_to_array(img)
X=X.reshape((1,)+X.shape)

i=0
for batch in datagen.flow(X,batch_size=1,save_to_dir='preview',save_prefix='B1',save_format='jpeg'):
    i+=1
    if i>20:
        break
