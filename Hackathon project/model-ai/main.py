
#pip install tensorflow
#pip install numpy
#pip install cv
#pip install opencv-python
#pip install Flask


import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import  VGG19


# tests for normal
vgg_model = VGG19(weights = 'imagenet',  include_top = False, input_shape = (180, 180, 3)) 

def predict_skin_disease(image_path):
    # Define list of class names
    class_names = ["Acne","Eczema","Atopic","Psoriasis","Tinea","vitiligo"]

    # Load saved model
    model = tf.keras.models.load_model('/Users/alice/6claass.h5')

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (180, 180))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = vgg_model.predict(img)
    img = img.reshape(1, -1)

    # Make prediction on preprocessed image
    print(class_names)
    pred = model.predict(img)[0]
    
    pred[0]=pred[0]*3
    print(pred)

    predicted_class_index = np.argmax(pred)
    predicted_class_second = np.argsort(pred)[-2]
 
    predicted_class_name = class_names[predicted_class_index]
    predicted_second_class_name = class_names[predicted_class_second]

    return predicted_class_name, predicted_second_class_name
