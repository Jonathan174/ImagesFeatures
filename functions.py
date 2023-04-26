import os
import csv
import pandas as pd
import cv2 


listaSubCarpetas=[]

def clasificarImagenes():
    # Ruta de la carpeta principal
    ruta_principal = 'Imagenes 2do proyecto'

    # Lista para almacenar los nombres de las imágenes
    lista_nombres_imagenes = []

    # Recorrer todas las subcarpetas y obtener los nombres de las imágenes
    for ruta, subcarpetas, archivos in os.walk(ruta_principal):
        nombre_subcarpeta=""
        for archivo in archivos:
            # Solo consideramos archivos de imagen
            if archivo.endswith('.jpg') or archivo.endswith('.png'):
                # Obtener el nombre completo del archivo
                ruta_archivo = os.path.join(ruta, archivo)

                # Dividir la ruta en cada carpeta y subcarpeta
                lista_carpetas = []
                while True:
                    ruta_archivo, carpeta = os.path.split(ruta_archivo)
                    if carpeta != "":
                        lista_carpetas.insert(0, carpeta)
                    else:
                        if ruta_archivo != "":
                            lista_carpetas.insert(0, ruta_archivo)
                        break

                # Obtener el nombre de la subcarpeta y el nombre del archivo
                nombre_subcarpeta = lista_carpetas[-2]
                nombre_archivo = lista_carpetas[-1]

                #Obtener la imagen con el metodo imread.
                image = cv2.imread('./Imagenes 2do proyecto/'+nombre_subcarpeta+'/'+nombre_archivo, cv2.IMREAD_COLOR)
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

                clasificacion = nombre_subcarpeta+";"+nombre_archivo+";"+str(meanR)+";"+str(meanG)+";"+str(meanB)+";"+str(meanGray)+";"+str(stdGray)
                # Añadir el nombre del archivo a la lista
                lista_nombres_imagenes.append(clasificacion)
        
        if len(nombre_subcarpeta)>0:
            listaSubCarpetas.append(nombre_subcarpeta)

    # Crear un DataFrame de Pandas con los nombres de las imágenes
    df = pd.DataFrame({'Nombre de imagen y vector de promedios': lista_nombres_imagenes})

    # Escribir el DataFrame en un archivo CSV
    df.to_csv('clasificacion imagenes.csv', index=False)


def validateInput(entrada, min, max):
    if entrada.isdigit():
        if int(entrada)<int(min) or int(entrada)>int(max):
            newEntrada = input("Escoja una opcion valida (Dentro del rango)\n")
            return validateInput(newEntrada, min, max)
        else:
            return entrada
    else:
        newEntrada = input("Escriba una opción valida, (numerica)\n")
        return validateInput(newEntrada, min, max)

def menuSubCarpetas():
    for i in range(len(listaSubCarpetas)):
        print("("+str(i+1)+")", listaSubCarpetas[i])
    opcion = input("Seleccione una subcarpeta\n")
    opcion2 =validateInput(opcion, 1, len(listaSubCarpetas))
    return menuImagenes(listaSubCarpetas[int(opcion2)-1])


def menuImagenes(subcarpeta):
    imageIndex = input("Select de index for the image do you want\n")    
    index2 =validateInput(imageIndex, 1, 100)
    return str(subcarpeta+"/"+index2+".png")


def menuOpciones():
    opcion = input("¿Que desea hacer?\nPresione 1 para seleccionar otra imagen\nPresione cualquier tecla para salir\n")
    return opcion