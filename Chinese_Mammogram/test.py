import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf


# Keras
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.backend import print_tensor

# Model saved with Keras model.save()
MODEL_PATH ='E:\ET-A\SDP\Chinese_Mammogram\malignancy_detector.model'

# Load your trained model
model = load_model(MODEL_PATH)

img1 = np.array(image.load_img("E:/ET-A/SDP/Chinese_Mammogram/Malignant/3.jpeg",target_size=(224,224)))
X1 = img1.reshape(1,224, 224, 3)
prediction=model.predict(X1)
predIdxs = np.argmax(prediction, axis=1)
print(predIdxs)