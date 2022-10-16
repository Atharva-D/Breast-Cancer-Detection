from __future__ import division, print_function
#import sys
import os
#import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='E:\ET-A\SDP\FINAL\Frontend\model_malignancy.h5'

# Load your trained model
model = load_model(MODEL_PATH)






def model_predict(filename,model):
    img1 = image.load_img(filename,target_size=(256,256))
    
    #plt.imshow(img1)
 
    Y = image.img_to_array(img1)
    
    X = np.expand_dims(Y,axis=0)
    val = model.predict(X)
    #print(val)
    if val == 1:
        
        preds = "The Severity of the Tissue is Malignant."
        
    
    elif val == 0:
        
        preds = "The Severity of the Tissue is Benign."

    else:
        preds = "It is not a Mammogram."

    return preds



@app.route('/', methods=['GET']) #tells application which URL should be called.
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
        
    return None


if __name__ == '__main__':
    app.run(port=5003,debug=True)
