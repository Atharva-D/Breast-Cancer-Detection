import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('E:\ET-A\SDP\Code\modelMNV2.h5')
modelMNV2 = tf.keras.models.load_model('modelMNV2.h5')



img1 = np.array(image.load_img("E:/ET-A/SDP/Code/all-mias/mdb042.pgm",target_size=(256,256)))
plt.imshow(img1)
X1 = img1.reshape(1,256, 256, 3)
predictionsMNV2=modelMNV2.predict(X1)
predictedMNV2 = [np.argmax(w) for w in predictionsMNV2]
#predictedMNV2=tuple(predictedMNV2)
print(predictedMNV2)
#0--Benign 1--Malignant



# img1 = np.array(image.load_img("E:/ET-A/SDP/Code/all-mias/mdb314.pgm",target_size=(256,256)))
# plt.imshow(img1)
# X1 = img1.reshape(1,256, 256, 3)
# predictionsMNV2=modelMNV2.predict(X1)
# predictedMNV2 = [np.argmax(w) for w in predictionsMNV2]
# print(predictedMNV2)
# #0--Benign 1--Malignant