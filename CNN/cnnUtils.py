# -*- coding: utf-8 -*-
"""
definir un fichier cnnUtil qui contient les fonctions(Utils) utilises par le cnn
"""
from keras.models import model_from_json

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

import cv2

# This function saves a model on the drive using two files : a json and an h5
def save_keras_model(model, filename):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename+".h5")
    
# This function loads a model from two files : a json and a h5
# BE CAREFUL : the model NEEDS TO BE COMPILED before any use !
def load_keras_model(filename):
    # load json and create model
    json_file = open(filename+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename+".h5")
    return loaded_model

# Evaluate a model using data and expected predictions
def print_model_error_rate(model, X_test, y_test):
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Model score : %.2f%%" % (scores[1]*100))
    print("Model error rate : %.2f%%" % (100-scores[1]*100))
    
# Save a data image to a real image on your desktop
def export_image_from_dataset(data, filename):
    im = Image.fromarray(data)
    im.save(filename)

   
   
# Charger une image et la convertir en tableau, pour une utilisation dans les modèles
def import_custom_image_to_dataset(filename):
    imgTraite = Image.open(filename).convert('L')
    #Resize like other images in dataset
    imgTraite = imgTraite.resize((28,28), Image.ANTIALIAS)
    #Convert to array
    x =  np.array(imgTraite)
    #Reshape
    x = x.reshape(1,28,28,1).astype('float32')
    #Normalize
    x = x / 255
        
    return x