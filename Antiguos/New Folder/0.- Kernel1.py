# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:25:05 2020

@author: raul
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Inicio de DataFrame
data = pd.DataFrame()
print(data)

#Inicio Lectura Camara
#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("C:/Users/raul/Desktop/8.- Tesis/5.- Redes Neuronales/Muestras/Muestras/Oficina/video11.mp4")

fourcc = cv2.VideoWriter_fourcc(*"XVID")
salida = cv2.VideoWriter("output.mp4",fourcc, 35, (int(cam.get(3)),int(cam.get(4))))

#Tamaño mira
rango = 40

#Valor Inicial Kernel
ker = 20

while(True):
    #Generación de kernel en blanco
    kernel = np.ones((ker,ker),np.uint16)

    #Lectura de frame
    ret,frame = cam.read()

    #Filtrado RGB
    filtroRGB = cv2.inRange(frame, np.array([240,240,240]), np.array([255,255,255]))

    #Transformación Morfológica Opening
    opening = cv2.morphologyEx(filtroRGB,cv2.MORPH_OPEN,kernel)

    #Generación Rectangulo y obtención de coordenadas
    x,y,w,h = cv2.boundingRect(opening)

    #Generación de punto central de la luz reconocida
    cv2.circle(frame,(int(x+w/2),int(y+h/2)),6,(0,0,100),-1)

    #Obtención de dimensiones del frame
    height, width = frame.shape[:2]

    #Dibujo de rectangulo central o "mira"
    cv2.rectangle(frame,(int(width/2-rango),int(height/2-rango)),(int(width/2+rango),int(height/2+rango)),(0,0,0),4)

    #Dibujo de rectangulo en haz de luz
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),4)

    #Recorte ROI
    if w>0:
        ROI = frame[y:y+h,x:x+w]
        cv2.imshow("ROI",ROI)
    
    #Control de Direccion
    if x+(w/2) > (width/2)+rango:
        print("<--")
    if x+(w/2) < (width/2)-rango:
        print("-->")
    if y+(h/2) > height/2+rango:
            print("^ \n|")
    if y+(h/2) < height/2-rango:
            print("  | \n  v")

    #Adaptación de kernel
    if (width/(w+1))+(height/(h+1)) < 5:
        ker = ker+1
    if w == 0:
        ker = 1

    #Obtención de Información Relevante
    df = pd.DataFrame([[x+(w/2) , y+(h/2) , w , h , w>0 , x , x+w , y , y+h ]], columns=["X_Central_Luz","Y_Central_Luz","Width_Luz","Height_Luz","ROI","Xi","Xf","Yi","Yf"])
    data = data.append(df)

    #Mostrar por pantalla
    cv2.imshow("camara",frame)
    #cv2.imshow("mascara",mascara)
    cv2.imshow("Morfologia",opening)
    salida.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
            break
data.to_csv("data.csv")
print(data)
cam.release()
#salida.release()
cv2.destroyAllWindows()