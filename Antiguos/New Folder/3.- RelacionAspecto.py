import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

#video = cv2.VideoCapture(0)
video = cv2.VideoCapture("C:/Users/raul/Desktop/8.- Tesis/5.- Redes Neuronales/Muestras/Muestras/Oficina/video11.mp4")
ker = 20
rango = 40
p = 95
p2 = 1-(p/100)
a = 0
_,frame = video.read()
cv2.imwrite("frame.jpeg",frame)
hf,wf,dimf = frame.shape
cv2.imshow("Video",frame)
while(True):
	_,frame = video.read()
	I = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	I = cv2.GaussianBlur(I,(11,11),0)
	umbral = np.max(I)*(p/100)
	mascara = np.uint8((I>umbral)*255)
	#mascara = cv2.resize(mascara,None,fx=0.6,fy=0.6)
	cv2.imshow("Mascara",mascara)

	#Componentes que se han encontrado en la mascara
	output = cv2.connectedComponentsWithStats(mascara,4,cv2.CV_32S)
	cantObj = output[0]
	labels = output[1]
	stats = output[2]
	#n = int(cantObj*p2)
	n = int(cantObj*0.5)
	#numeros = np.arange(1,n,1)
	numeros = np.arange(1,cantObj,1)
	ocupacion_en_porcentaje_del_frame = 3
	a += 1
	#for i in numeros:
		#x = stats[:][i][0]
		#y = stats[:][i][1]
		#h = stats[:][i][2]
		#w = stats[:][i][3]

		#if w*100/wf >= ocupacion_en_porcentaje_del_frame or h*100/hf >= ocupacion_en_porcentaje_del_frame:
			#recorte = frame[y:y+w,x:x+h]
			#cv2.imwrite("./recortes/emisor"+str(a)+"-"+str(i)+".jpeg",recorte)
			#cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2)

		#if recorte.any() == True:
			#cv2.imshow("Captura"+str(i),recorte)
			#print(len(recorte))
			#cv2.imwrite("./recortes/emisor"+str(a)+"-"+str(i)+".jpeg",recorte)


	contours, hierarchy	= cv2.findContours(mascara,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for i in contours:
		rect = cv2.minAreaRect(i)
		centro, dimensiones, rotacion = cv2.minAreaRect(i)
		#print(rect)
		#print(rect[0])
		#print(rect[1])
		#print(rect[2])
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		print(box)
		print(box[0][0])
		print(box[0][1])
		print(box[1][0])
		print(box[1][1])
		print(box[2][0])
		print(box[2][1])
		print(box[3][0])
		print(box[3][1])
		#print(dimensiones[0])
		#print(dimensiones[1])
		#print(dimensiones[0]*dimensiones[1])
		area_cantidad_de_pixels_minimo = 10
		if dimensiones[0]*dimensiones[1] >= area_cantidad_de_pixels_minimo:
		#if dimensiones[0]*dimensiones[1] >=
			if abs(dimensiones[0]/(dimensiones[1]+1)) >= 0.2:
				im = cv2.drawContours(frame,[box],0,(0,0,255),2)
			if abs(dimensiones[0]/(dimensiones[1]+1)) <= 0.2:
				im = cv2.drawContours(frame,[box],0,(0,255,0),2)




		#cv2.circle(frame,(int(rect[0][0]),int(rect[0][1])),3,(0,0,100),-1)
		
		#print(contours)
		#print(i)
		#im = cv2.drawContours(frame,contours,2,(0,0,255),2)
		#cv2.imshow("ima",im)

	
	
	"""
	for i in contours:
			rect = cv2.boundingRect(i)
			x,y,w,h = rect
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	"""
	if isinstance(frame,np.ndarray):
		frame = cv2.resize(frame,None,fx=0.6,fy=0.6)
		#mascara = cv2.resize(mascara,None,fx=0.8,fy=0.8)
		cv2.imshow("Video",frame)
		#cv2.imshow("Mascara",mascara)
	if cv2.waitKey(1) & 0xFF == ord('s'):
		break
video.release()
cv2.destroyAllWindows()
