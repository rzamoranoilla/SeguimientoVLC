import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

#video = cv2.VideoCapture(0)
video = cv2.VideoCapture("C:/Users/raul/Desktop/8.- Tesis/5.- Redes Neuronales/Muestras/Muestras/Oficina/video11.mp4")
#video = cv2.VideoCapture("http://ti.smartpartners.cl/808gps/open/player/video.html?lang=en&vehiIdno=CAMdiam&account=Raul&password=Seguridad2021&channel=1")
ker = 20
rango = 40
p = 95
p2 = 1-(p/100)
a = 0
_,frame = video.read()
hf,wf,dimf = frame.shape
cv2.imshow("Video",frame)
while(True):
	_,frame = video.read()
	I = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	umbral = np.max(I)*(p/100)
	mascara = np.uint8((I>umbral)*255)

	#Componentes que se han encontrado en la mascara
	output = cv2.connectedComponentsWithStats(mascara,4,cv2.CV_32S)
	cantObj = output[0]
	labels = output[1]
	stats = output[2]
	#print(stats)
	#print(len(stats))
	#print(int(cantObj*0.6))
	#n = int(cantObj*p2)
	n = int(cantObj*0.5)
	a += 1
	numeros = np.arange(1,n,1)
	b = 0
	for i in numeros:
		x = stats[:][i][0]
		y = stats[:][i][1]
		h = stats[:][i][2]
		w = stats[:][i][3]
		
		#print(i,x,y,w,h)
		recorte = frame[x:x+h,y:y+w]

		wrec,hrec,dimrec = recorte.shape
		#print(wrec,hrec,dimrec)

		print(w-wrec)

		if w/wf >= 0.02:
			print(b)
			b+=1
			cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2)

		#if recorte.any() == True:
			#cv2.imshow("Captura"+str(i),recorte)
			#print(len(recorte))
			#cv2.imwrite("./emisor"+str(a)+"-"+str(i)+".jpeg",recorte)


	"""
	print("asd" ,cantObj)
	print(stats[:][1])
	print(stats[:][1][0])
	print(stats[:][1][1])
	print(stats[:][1][2])
	print(stats[:][1][3])
	x = stats[:][1][0]
	y = stats[:][1][1]
	w = stats[:][1][2]
	h = stats[:][1][3]
	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	"""

	#for i in 

	#print(stats[:][1][0])

	#contours, hierarchy	= cv2.findContours(mascara,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	"""
	for i in contours:
			rect = cv2.boundingRect(i)
			x,y,w,h = rect
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	"""
	if isinstance(frame,np.ndarray):
		frame = cv2.resize(frame,None,fx=0.8,fy=0.8)
		mascara2 = cv2.resize(mascara,None,fx=0.8,fy=0.8)
		cv2.imshow("Video",frame)
		cv2.imshow("Mascara",mascara)
	if cv2.waitKey(1) & 0xFF == ord('s'):
		break
video.release()
cv2.destroyAllWindows()
