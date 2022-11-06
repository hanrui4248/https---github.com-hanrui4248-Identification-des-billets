"""
definir une interface d'utilisateur, pour l'dentification de la dénomination des billets de banque
"""
from faulthandler import disable
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import traiter_image.extractionDesChiffres as ex
import CNN.cnn as cnn

class interface(tk.Tk):
    #constructeur
    def __init__(self):
        tk.Tk.__init__(self)
        self.createWidget()

    #creer les composants d'interface utilisateur
    def createWidget(self):
        # definir les variables d'instances
        self.photo = None
        self.img_path = None

        self.labelImg = tk.Label(self, image=self.photo,relief="ridge", bd = 5)
        self.labelMsg= tk.Label(self, text="Vous n'avez sélectionné aucune image",font=('Arial', 20), width=51, height=12) #relief="raised"
        self.labelMsg.place(x=93, y=320, anchor='sw')
        self.l = tk.Label(self, text='Résultats de la \n classification:', font=('Arial', 20))
        self.l.place(x=180, y=430, anchor='sw')
        #definir un entry pour afficher le resultat de la classification
        self.e = tk.Entry(self, show=None, font=('Arial', 60),relief="ridge", bd = 5,width=10)
        self.e.place(x=340, y=460, anchor='sw')
        self.btn01 = tk.Button(self, text='Ouvrir une image', command= self.getImg, bg='white',height=2)
        self.btn01.place(x=110, y=600, anchor='sw')
        self.btn02 = tk.Button(self, text="Classifier les dollars \ncanadiens", command= self.classifierImageBillet, bg='white')
        self.btn02.config(state = 'disabled')
        self.btn02.place(x=265, y=600, anchor='sw')
        self.btn03 = tk.Button(self, text="Afficher le processus\n d'extraction", command= self.afficherProcessusExtraction, bg='white' ,height=2)
        self.btn03.config(state = 'disabled')
        self.btn03.place(x=435, y=600, anchor='sw')
        self.btn04 = tk.Button(self, text="Quitter", command= quit, bg='white' ,height=2)
        self.btn04.place(x=610, y=600, anchor='sw')

    # definir une methode pour lire une image a partir d'une repertoire et l'afficher dans le label.
    def getImg(self):
        # lire une image a partir d'une repertoire
        self.img_path = filedialog.askopenfilename(title='choisir un fichier')
        img = Image.open(self.img_path)
        width, height = img.size
        img =  img.resize((600,272))
        #effacer le label de message
        self.labelMsg.place_forget()
      
        self.labelImg.place(x=93, y=320, anchor='sw')
        # afficher l'image dans le label  
        self.photo = ImageTk.PhotoImage(img)  
        self.labelImg.configure(image = self.photo)
        self.labelImg.image = self.photo

        self.btn03.config(state = 'disabled')
        self.btn02.config(state = 'active')
        self.e.delete(0, 'end')
    

    # une methode qui afficher la valeur du billet
    def classifierImageBillet(self):
        if(self.img_path!=None):
            ex.extractionChiffres(self.img_path)
            resultats = cnn.classifierLesChiffre()
            valeurBillet = ''
            if(len(resultats)!=0):
                for x in range(len(resultats)):
                    if(resultats[x]==1 or resultats[x]==2 or resultats[x]== 5):
                        valeurBillet = str(resultats[x]) + valeurBillet
                    elif(resultats[x]==0):
                        valeurBillet = valeurBillet + '0'
                    else:
                        valeurBillet = 'Echec de la classification!'
                        break
            else:
                valeurBillet = 'Echec de la classification!'
        
        if(valeurBillet != 'Echec de la classification!'):
            valeurBillet += 'CAD'

        self.e.insert(0,valeurBillet)
        self.btn03.config(state = 'active')
        self.btn02.config(state = 'disabled')

    

    # une methode qui afficher le processus d'identification des chiffres des billets 
    def afficherProcessusExtraction(self):
        ex.afficher_images_triate()


#fonction main
if __name__ == "__main__":
    appli_phase1 = interface()
    appli_phase1.title("TRAVAIL DE SESSION")
    appli_phase1.geometry('800x680')  
    appli_phase1.mainloop()
    
