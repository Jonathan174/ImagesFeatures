import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
#%matplotlib inline
from functions import *

def showImage(imageOriginal, imageGrayScale) :
    cv2.imshow("Original", imageOriginal)
    cv2.imshow("Gray Scale", imageGrayScale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

clasificarImagenes()
step = "1"
while(step=="1"):
    ruta=menuSubCarpetas()
    #Obtener la imagen con el metodo imread.
    image = cv2.imread('./Imagenes 2do proyecto/'+ruta, cv2.IMREAD_COLOR)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Dividir la imagen en canales RGB
    b, g, r = cv2.split(image)
    # Calcular la media de cada canal
    meanB = round(cv2.mean(b)[0], 4) 
    meanG = round(cv2.mean(g)[0], 4)
    meanR = round(cv2.mean(r)[0], 4)
    meanGray, stdGray = cv2.meanStdDev(grayImage)     #Desviación estandar
    meanGray = round(meanGray[0][0], 4)
    stdGray = round(stdGray[0][0], 4)
    print("Archivo: ", ruta)
    print("Promedio de Red: ",meanR, "Promedio de Green: ",meanG, "Promedio de Blue: ",meanB, "Promedio de Gray: ",meanGray, "Desviación estandar: ",stdGray)
    showImage(image, grayImage)
    #showImage(grayImage)
    plt.show()

    step = menuOpciones()