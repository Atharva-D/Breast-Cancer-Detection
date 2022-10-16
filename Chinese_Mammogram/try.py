import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory("E:/ET-A/SDP/Chinese_Mammogram/CM/Train/",
                                          target_size=(224,224),
                                          batch_size = 32,
                                          class_mode = 'binary')
                                         
test_dataset = test.flow_from_directory("E:/ET-A/SDP/Chinese_Mammogram/CM/Test/",
                                          target_size=(224,224),
                                          batch_size =32,
                                          class_mode = 'binary')

print(test_dataset.class_indices)


model = keras.Sequential()

# Convolutional layer and maxpool layer 1
model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 2
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 3
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 4
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# This layer flattens the resulting image array to 1D array
model.add(keras.layers.Flatten())

# Hidden layer with 512 neurons and Rectified Linear Unit activation function 
model.add(keras.layers.Dense(512,activation='relu'))

# Output layer with single neuron which gives 0 for Cat or 1 for Dog 
#Here we use sigmoid activation function which makes our model output to lie between 0 and 1
model.add(keras.layers.Dense(1,activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#steps_per_epoch = train_imagesize/batch_size

model.fit(train_dataset,
         steps_per_epoch = 250,
         epochs = 5,
         validation_data = test_dataset
       
         )


model.save("Cancer_detector.model", save_format="h5")

def predict_Severity(filename):
    img1 = image.load_img(filename,target_size=(224,224))
    
    plt.imshow(img1)
    plt.plot(img1)
    plt.show()
 
    Y = image.img_to_array(img1)
    
    X = np.expand_dims(Y,axis=0)
    val = model.predict(X)
    print(val)
    if val == 1:
        
        plt.xlabel("MALIGNANT",fontsize=30)
        
    
    elif val == 0:
        
        plt.xlabel("BENIGN",fontsize=30)


predict_Severity("E:\ET-A\SDP\Chinese_Mammogram\Malignant\M_0_75.jpeg")