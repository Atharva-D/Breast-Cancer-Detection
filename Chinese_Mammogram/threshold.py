import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, img_as_ubyte
from skimage.filters import threshold_multiotsu
from tensorflow.python.keras.preprocessing.image import img_to_array


da=[]
image = io.imread("E:/ET-A/SDP/FINAL/Chinese_Mammogram/Malignant/5.jpeg")
threshold=threshold_multiotsu(image,classes=4)
regions=np.digitize(image,bins=threshold)
output=img_as_ubyte(regions)
#plt.imsave("Plots_final/Otsu_segmented.jpeg",output)
plt.imshow(output)
plt.plot(output)
plt.show()
print(image)
print(output)
#im=img_to_array(image)
