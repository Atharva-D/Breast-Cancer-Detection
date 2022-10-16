import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import glob
import io
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

Benign_Images=glob.glob("E:/ET-A/SDP/Chinese_Mammogram/Benign/*.jpeg")
Benign_Images_1=glob.glob("E:/ET-A/SDP/Chinese_Mammogram/Benign/*.jpg")
#print(len(Benign_Images_1))
#print(len(Benign_Images))
Benign_Images.extend(Benign_Images_1)
print(len(Benign_Images))

Malignant_Images=glob.glob("E:/ET-A/SDP/Chinese_Mammogram/Malignant/*.jpeg")
print(len(Malignant_Images))


datagen = ImageDataGenerator(validation_split=0.20, height_shift_range=0.10,width_shift_range=0.10,rotation_range=30,rescale=1/255.)

for i in Malignant_Images:
    img=load_img(i,target_size=(224, 224))
    X=img_to_array(img)
    X=X.reshape((1,)+X.shape)
    j=0
    for batch in datagen.flow(X,batch_size=1,save_to_dir='preview_Malignant',save_prefix='M',save_format='jpeg'):
        j+=1
        if j>20:
            break