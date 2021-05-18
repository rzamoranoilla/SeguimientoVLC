import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy import ndimage
import time
import pandas as pd

video = cv2.VideoCapture("C:/Users/raul/Desktop/8.- Tesis/5.- Redes Neuronales/Muestras/Muestras/Oficina/video11.mp4")
#video = cv2.VideoCapture(0)
#time.sleep(1)

def dist(x1,y1,x2,y2):
	dist = np.sqrt((x2-x1)**2+(y2-y1)**2)
	return dist

data2 = pd.DataFrame()
print(data2)

ker = 20
rango = 40
p = 95
p2 = 1-(p/100)
a = 0
_,frame = video.read()
hf,wf,dimf = frame.shape
umbral2 = 10
etiquetas = []
datos = []
while(True):
	#time.sleep(1)
	_,frame = video.read()
	I = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	I = cv2.GaussianBlur(I,(11,11),0)
	umbral = np.max(I)*(p/100)
	mascara = np.uint8((I>umbral)*255)

	output = cv2.connectedComponentsWithStats(mascara,4,cv2.CV_32S)
	cantObj = output[0]
	labels = output[1]
	stats = output[2]
	n = int(cantObj*0.5)
	numeros = np.arange(1,cantObj,1)
	ocupacion_en_porcentaje_del_frame = 3

	canny = cv2.Canny(mascara,20,150)
	contornos,  hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	for i in contornos:
		rect = cv2.minAreaRect(i)
		centro, dimensiones, rotacion = cv2.minAreaRect(i)
		#print("dimensiones_minimas",dimensiones)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		area_cantidad_de_pixels_minimo = 10
		if dimensiones[0]*dimensiones[1] >= area_cantidad_de_pixels_minimo:
			tasa = float(dimensiones[1])/float(dimensiones[0]) if dimensiones[1]>dimensiones[0] else float(dimensiones[0])/float(dimensiones[1])
			#print("tasa",tasa)

		epsilon = 0.01*cv2.arcLength(i,True)
		approx = cv2.approxPolyDP(i,epsilon,True)
		x,y,w,h = cv2.boundingRect(approx)

		dimensiones2 = (w,h)
		#print("Dimensiones_rectas", dimensiones2)

		l = 9
		if len(approx) >= l:
			cv2.drawContours(frame, approx, -1, (0, 0, 255), 2, cv2.LINE_AA)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255), 2)
			etiqueta = "Emisor"
		if len(approx) < l:
			cv2.drawContours(frame, approx, -1, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255 ,0), 2)
			etiqueta = "No-Emisor"

		cantidad_aristas = len(approx)
		#print("Cantidad_Aristas",cantidad_aristas)
		cv2.drawContours(frame, [approx], 0, (255,0,0),2)

		df = pd.DataFrame([[dimensiones,dimensiones2,tasa,cantidad_aristas,etiqueta]], columns=["Dimensiones_minimas","Dimensiones_rectas","Tasa","Cantidad_Aristas","Etiqueta"])
		data2 = data2.append(df)
		datos.append([dimensiones[0]*dimensiones[1],dimensiones2[0]*dimensiones2[1],tasa,cantidad_aristas])
		etiquetas.append(etiqueta)

	frame = cv2.resize(frame,None,fx=0.6,fy=0.6)
	cv2.imshow("Frame",frame)

	if cv2.waitKey(1) & 0xFF == ord('s'):
		break
data2.to_csv("data2.csv")
#print(datos)
#print(etiquetas)




datos = np.array(datos)
etiquetas = np.array(etiquetas)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

for i in range(0,len(etiquetas)):
	if etiquetas[i] == "Emisor":
		ax.scatter(datos[i,0],datos[i,2],datos[i,3],marker = "*",c = 'y')
	else:
		ax.scatter(datos[i,0],datos[i,2],datos[i,3],marker = "^",c = 'r')
ax.set_xlabel('Dimensiones_rectas')
ax.set_ylabel('Tasa')
ax.set_zlabel('Cantidad_Aristas')

A = np.zeros((4,4))

b = np.zeros((4,1))

#datos = [datos]
print(datos)
#datos = np.append([1],datos[1])
#print(np.shape(datos[1]))
#datos = np.append([1], datos[1])
#print(np.shape(datos[1]))

"""
for i in range(0,len(etiquetas)):
	print(datos[i])
	print(np.shape(datos[i]))
	x = np.append([1],[datos[i]])
	print(np.shape(x))
	#print(x)
	x = x.reshape((4,1))
	y = etiquetas[i]
	A = A+x*x.T
	b = b+x*y
inv = np.linalg.inv(A)
w = np.dot(inv,b)
X = np.arrange(0,1,0.1)
Y = np.arrange(0,1,0.1)
X,Y = np.meshgrid(X,Y)
Z = -(w[0]+w[1]*X+w[2]*Y)/w[3]
surf = ax.plot_surface(X,Y,Z,cmap= cm.Blues)
"""
plt.show()




#etiquetas.to_csv("etiquetas.csv")
#print(data)
video.release()
cv2.destroyAllWindows()