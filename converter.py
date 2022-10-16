import pydicom
import os
import numpy as np
from PIL import Image
import glob



Images=glob.glob("E:/ET-A/SDP/dcm/DCM/*.dcm")
#print(Images)
for i in Images:
    ds = pydicom.dcmread(i)
    new_image = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    #final_image.show()
    final_image.save(i+'.jpeg')
    final_image.copy()
    

   


# DCM='E:\ET-A\SDP\dcm\DCM'
# for name in DCM:
    # ds = pydicom.dcmread(name)
    # new_image = ds.pixel_array.astype(float)
    # scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    # scaled_image = np.uint8(scaled_image)
    # final_image = Image.fromarray(scaled_image)
    # final_image.show()
    # final_image.save(name+'.jpeg')


# def get_names(path):
#     names = []
#     for root, dirnames, filenames in os.walk(path):
#         for filename in filenames:
#             _, ext = os.path.splitext(filename)
#             if ext in ['.dcm']:
#                 names.append(filename)
    
#     return names

# def convert_dcm_jpg(name):
    
#     im = pydicom.dcmread('Database/'+name)

#     im = im.pixel_array.astype(float)

#     rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
#     final_image = np.uint8(rescaled_image) # integers pixels

#     final_image = Image.fromarray(final_image)

#     return final_image


# DCM='DCM'
# names = get_names(DCM)
# for name in names:
#     image = convert_dcm_jpg(name)
#     image.save(name+'.jpeg')