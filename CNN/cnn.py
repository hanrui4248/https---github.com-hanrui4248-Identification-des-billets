# -*- coding: utf-8 -*-
"""
definir un ficher cnn qui est capable de reconnaitre les chiffre entre 0-9
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Softmax
from keras.layers.convolutional import Conv2D
from CNN import preparedata as pr
from CNN import cnnUtils as cu
import os

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# import MNIST dataset
(X_train, y_train), (X_test, y_test), num_classes = pr.get_and_prepare_data_mnist()

# define the small model
def small_model():
    # create model
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Identifiez les chiffres représentés par toutes les images du repertoire, stockez les résultats  dans une liste et le retourner
def classifierLesChiffre():
    # build the model
    # model = small_model()
    # Fit the model
    # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

    #Afin de gagner du temps d'exécution, j'ai déjà formé le modèle cnn que j'ai créé à l'avance, il me suffit donc de charger le modèle cnn formé et de le compiler
    model = cu.load_keras_model("CNN/save_model/small_model_cnn")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Save the model
    #cu.save_keras_model(model, "CNN/save_model/small_model_cnn")

    #Cette étape renvoie l'ensemble d'étiquettes prédit
    probability_model = Sequential([model, Softmax()])
    #definir une liste qui stocker les resultats de la claasification
    resultats = []
    #L'image à reconnaître est stockée dans la repertoire traiter_image/imgs_roi
    filelist=os.listdir('traiter_image/imgs_roi')
    for fileName in filelist:
        if(fileName.endswith('.jpg')):
            #parcourir chaque image dans le reprtoire , et faire la classification
            img = cu.import_custom_image_to_dataset('traiter_image/imgs_roi/'+fileName)
            predictions_single = probability_model.predict(img)
            #La variable resultat est le résultat de la classification d'une seule image
            resultat = np.argmax(predictions_single[0]) 
            """
            Il y a deux défauts dans le modèle cnn. Le premier est qu'il ne peut pas reconnaître le chiffre 0 à 100 %. 
            Après mes tests, il reconnaît parfois le 0 du billet comme un chiffre tel que 8 ou 9. 
            Le deuxième défaut est que le modèle ne peut reconnaître que le chiffre 1 sous la forme d'une barre verticale, 
            et le chiffre 1 sur les billets sont toujours reconnus comme 7.
            Afin d'obtenir des résultats corrects, j'ai décidé de corriger ces erreurs manuellement
            """
            if(resultat !=5 and resultat !=2 and resultat !=0 and resultat!=7): 
                resultat = 0
            elif(resultat == 7):
                resultat = 1
            #ajouter les resultats
            resultats.append(resultat) 
    # print(resultats)
    return resultats

