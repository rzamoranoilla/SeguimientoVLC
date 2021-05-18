import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time

video = cv2.VideoCapture("C:/Users/raul/Desktop/8.- Tesis/5.- Redes Neuronales/Muestras/Muestras/Oficina/video11.mp4")
#video = cv2.VideoCapture(0)

p = 95
#time.sleep(5)

while(True):
	_,frame = video.read()
	#frame = cv2.resize(frame,None,fx=0.2,fy=0.2)
	I = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	I = cv2.GaussianBlur(I,(11,11),0)
	umbral = np.max(I)*(p/100)
	mascara = np.uint8((I>umbral)*255)
	#mascara2 = cv2.resize(mascara,None,fx=0.6,fy=0.6)
	#cv2.imshow("Mascara",mascara2)

	canny = cv2.Canny(mascara,20,150)
	#canny2 = cv2.resize(canny,None,fx=0.6,fy=0.6)
	canny2 = canny.copy()
	canny2 = ndimage.binary_fill_holes(canny2).astype(np.uint8)*255
	contornos,  _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contornos,  _ = cv2.findContours(canny2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	canny2 = cv2.cvtColor(canny2,cv2.COLOR_GRAY2RGB)
	cv2.drawContours(canny2, contornos, -1, (0, 0, 255), 2, cv2.LINE_AA)
	cv2.imshow("Bordes",canny2)
	#cv2.drawContours(frame, contornos, -1, (0, 0, 255), 2, cv2.LINE_AA)
	#frame = cv2.resize(frame,None,fx=0.6,fy=0.6)
	#cv2.imshow("Frame",frame)

	for i in contornos:
		epsilon = 0.01*cv2.arcLength(i,True)
		approx = cv2.approxPolyDP(i,epsilon,True)
		x,y,w,h = cv2.boundingRect(approx)
		l = 9
		if len(approx) >= l:
			cv2.drawContours(frame, approx, -1, (0, 0, 255), 2, cv2.LINE_AA)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
		if len(approx) < l:
			cv2.drawContours(frame, approx, -1, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

		#cv2.drawContours(frame, approx, -1, (0, 0, 255), 2, cv2.LINE_AA)
		print(len(approx))
		cv2.drawContours(frame, [approx], 0, (255,0,0),2)

	frame = cv2.resize(frame,None,fx=0.6,fy=0.6)
	cv2.imshow("Frame",frame)



	if cv2.waitKey(1) & 0xFF == ord('s'):
		break
video.release()
cv2.destroyAllWindows()