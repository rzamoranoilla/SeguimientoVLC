import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

video = cv2.VideoCapture(0)
#video = cv2.VideoCapture("video11.mp4")
#video = cv2.VideoCapture("http://ti.smartpartners.cl/808gps/open/player/video.html?lang=en&vehiIdno=CAMdiam&account=Raul&password=Seguridad2021&channel=1")
ker = 20
rango = 40
p = 95
#frame = read.
while(True):
	_,frame = video.read()
	kernel = np.ones((ker,ker),np.uint16)
	I = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	umbral = np.max(I)*(p/100)
	print(umbral)
	mascara = np.uint8((I>umbral)*255)
	contours, hierarchy	= cv2.findContours(mascara,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for i in contours:
			rect = cv2.boundingRect(i)
			x,y,w,h = rect
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	if isinstance(frame,np.ndarray):
		frame = cv2.resize(frame,None,fx=0.8,fy=0.8)
		mascara = cv2.resize(mascara,None,fx=0.8,fy=0.8)
		cv2.imshow("Video",frame)
		cv2.imshow("Mascara",mascara)
	if cv2.waitKey(1) & 0xFF == ord('s'):
		break
video.release()
cv2.destroyAllWindows()
