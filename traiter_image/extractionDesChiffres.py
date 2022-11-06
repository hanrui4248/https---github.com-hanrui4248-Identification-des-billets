"""
l'extraction des chiffres sur un billet
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil

#Ces deux  constants sont les rapports de toutes les longueurs et largeurs possibles des chiffres sur les billets !
PROPORTION1 = float(54/78)
PROPORTION2 = float(30/87) 
#Erreur maximale autorisée
EPSILON = 0.03 
#La plage de valeurs de la longueur et de la largeur des chiffres sur les billets
WIDTH_MIN = 29
WIDTH_MAX = 63
HEIGHT_MIN = 77
HEIGHT_MAX = 91
#definir une variable global qui stocke les images traites
images = []


#cette methode peut extraire les chiffres sur l'image du billet , recadrer l'image traitée, ne conserver que la zone représentant un seul chiffre 
# et enregistrer l'image recadrée dans le repetoire images_roi
def extractionChiffres(path):

    #lire et resize l'image 
    im = cv2.imread(path)
    im = cv2.resize(im, (600,272)) 
    img_orginal = im.copy()

    kernel = np.ones((7,7),np.uint8)
    #transformer l'image en niveaux de gris
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #Appliquer un lissage gaussien en utilisant un noyau 5×5 pour réduire le bruit haute fréquence.
    imgLisse = cv2.GaussianBlur(imgray, (5, 5), 0) 
    #Utilisez le seuillage 
    ret3,th3 = cv2.threshold(imgLisse,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    #Applique la fermeture
    imgFermeture = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel) 
    #Trouver tous les contours apres fermeture
    contours, hierarchy = cv2.findContours(imgFermeture,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contoursChiffres = []
    #Selon le rapport entre la longueur et la largeur du contour et la plage de longueur et de largeur, le contour représentant le nombre est filtré
    for i in range(len(contours)):
        temp = contours[i]
        x,y,w,h = cv2.boundingRect(temp)
        if((abs(w/h-PROPORTION1)<=EPSILON or abs(w/h-PROPORTION2)<=EPSILON) and (w>=WIDTH_MIN and w<=WIDTH_MAX) and (h>=HEIGHT_MIN and h<=HEIGHT_MAX)):
            contoursChiffres.append(temp)
    
    #vider la reprtoire imgs_roi
    imgsRoisFolderPath = 'traiter_image/imgs_roi'
    shutil.rmtree(imgsRoisFolderPath)
    os.mkdir(imgsRoisFolderPath)

    # Recadrez la zone de l'image qui représente le chiffre et stockez-la dans la repertoire imgs_roi
    for j in range(len(contoursChiffres)):
        x,y,w,h = cv2.boundingRect(contoursChiffres[j])
        #Encerclez la zone représentant les chiffres dans l'image originale
        imgAvecRoi = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        #Recadrez les zones qui représentent des chiffres
        imgRoi=imgFermeture[y:y+h,x:x+w]
        imgRoiPath = imgsRoisFolderPath+'/imgRoi'+str(j)+'.jpg'
        cv2.imwrite(imgRoiPath,imgRoi)
    
    #si on a pas trouve les chiffres
    if(len(contoursChiffres)==0):
        imgAvecRoi = img_orginal

    global images
    images = [img_orginal,imgLisse,th3,imgFermeture,imgAvecRoi]

    


#Cette fonction peut afficher le changement de l'image à chaque étape du processus de reconnaissance des chiffres et afficher les images recadrées
def afficher_images_triate():
    global images
    b,g,r = cv2.split(images[0]) 
    images[0] = cv2.merge([r,g,b]) 
    b,g,r = cv2.split(images[4]) 
    images[4] = cv2.merge([r,g,b]) 

    titles = ['1.img originale','2.lissage','3.seuillage',"4.fermeture","5.l'extraction des chiffres"]
    nomsImgRoi = os.listdir('traiter_image/imgs_roi')
    for i in range(5+len(nomsImgRoi)): 
        if(i<5):
            plt.subplot(3,3,i+1),plt.imshow(images[i],'gray') 
            plt.title(titles[i]) 
            plt.xticks([]),plt.yticks([])
        else:
            plt.subplot(3,3,i+1),plt.imshow(cv2.imread('traiter_image/imgs_roi/'+nomsImgRoi[i-5]),'gray') 
            plt.title('images recadree '+str(i-4)) 
            plt.xticks([]),plt.yticks([])

    plt.show()








