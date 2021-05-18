import cv2
import numpy as np
import funciones as fn
import math
import time

#img = cv2.imread("C:/Users/raul/Desktop/8.- Tesis/5.- Redes Neuronales/Muestras/Muestras/COCO/emisores/000000000032.jpg")
img = cv2.imread("C:/Users/raul/Desktop/8.- Tesis/5.- Redes Neuronales/Muestras/Muestras/Images/img5.jpeg")
p_umbral = 90

imagen = img.copy()
mascara = fn.segmentador_de_histograma(img,p_umbral)
contornos, hierarchy = cv2.findContours(mascara,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagen, contornos, -1, (0, 0, 255), 2, cv2.LINE_AA)
for i in contornos:
	if len(i) >= 5:
		ellipse = cv2.fitEllipse(i)
		xcentroEllipse = ellipse[0][0]
		ycentroEllipse = ellipse[0][1]
		anchoEllipse = ellipse[1][0]
		altoEllipse = ellipse[1][1]
		rotationEllipse = ellipse[2]
		rotation = fn.translateRotation(rotationEllipse, anchoEllipse, altoEllipse)
		if (math.isnan(xcentroEllipse)): print("Nan")
		elif (math.isnan(ycentroEllipse)): print("Nan")
		elif (math.isnan(altoEllipse)): print("Nan")
		elif (math.isnan(anchoEllipse)): print("Nan")
		elif (math.isnan(rotationEllipse)): print("Nan")
		else:
			cv2.ellipse(imagen, ellipse, (0,255,0), 3)
			print("Pintado")
	else: print("Se necesitan mas puntos")

while(True):
	#imagen = cv2.resize(img, None, fx=0.8, fy=0.8)
	cv2.imshow("Img",img)
	cv2.imshow("Mascara",mascara)
	cv2.imshow("Imagen",imagen)
	if cv2.waitKey(1) & 0xFF == ord('s'):
		cv2.destroyAllWindows()
		break