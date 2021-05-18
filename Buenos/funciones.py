import cv2
import numpy as np
from scipy import ndimage
from tkinter import *
from skimage.measure import label
import math
import pandas as pd

def segmentador_de_histograma(frame,p_umbral):
	#De RGB a escala de grises
	I = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	#Umbral de intensidad a partir de p
	umbral = np.max(I)*(p_umbral/100)
	#Con mascara con condicionante obtenemos una mascara binarizada
	mascara = np.uint8((I>umbral)*255)
	return mascara

def relleno(frame): return ndimage.binary_fill_holes(frame).astype(np.uint8)*255

def ruido_gaussiano(frame,p_noise_shape):
	#Gaussian_kernel_shape
	gks = int((p_noise_shape/100)*frame.shape[0])
	#Gks debe ser impar
	if gks%2 == 0: gks += 1
	#Gaussian_Blur
	frame_blur = cv2.GaussianBlur(frame,(gks,gks),0)
	return frame_blur

def enmarcar_objetos(frame,mascara,dibujo):
	#Encontramos los contornos
	contours, hierarchy	= cv2.findContours(mascara,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	#Encerramos los contornos en rectangulos
	cajas_originales = np.empty((1, 4),dtype = np.int16)
	cajas_redimensionadas = np.empty((1, 4),dtype = np.int16)
	for i in contours:
			rect = cv2.boundingRect(i)
			x,y,w,h = rect
			#Dibujamos los rectangulos en el frame ingresado en un area mayor para contener el emisor y sus bordes
			#El area se incrementa en 1/3 de las dimensiones del lado del rectangulo original
			x1,y1,w1,h1 = int(x-(w/3)),int(y-(h/3)),int(5*w/3),int(5*h/3)
			#Agregamos el contorno a la lista
			cajas_originales = np.append(cajas_originales,[[x,y,w,h]],axis=0)
			cajas_redimensionadas = np.append(cajas_redimensionadas,[[x1,y1,w1,h1]],axis=0)
			if dibujo != 0:
				if len(str(dibujo)) >= 3 and str(dibujo)[0]==str(1):
					#Contornos
					cv2.drawContours(frame, contours, -1, (255, 0, 0), 2, cv2.LINE_AA)
				if len(str(dibujo)) >= 2 and str(dibujo)[1]==str(1):
					#Original
					cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
				if len(str(dibujo)) >= 1 and str(dibujo)[2]==str(1):
					#Redimensionado
					cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
	cajas_originales[0] = [0,0,10,10]
	cajas_redimensionadas[0] = [0,0,10,10]
	return frame,cajas_originales,cajas_redimensionadas,contours

def recorte(frame,x,y,w,h): return frame[y:y+h,x:x+w]

def ventana_categoria():
	def emisor():
	    cat.set("Emisor")
	    root.config(background = "green")

	def reflejo():
		cat.set("Reflejo")
		root.config(background = "yellow")

	def emisor_ambiental():
		cat.set("Emisor_Ambiental")
		root.config(background = "white")

	def no_emisor():
	    cat.set("No_Emisor")
	    root.config(background = "red")

	def listo():
		print(cat.get)
		if cat.get() == "":
			print("No ha ingresado valores.")
		else:	
			estado.set(True)
			root.destroy()

	def omitir():
		estado.set(False)
		root.destroy()

	def finalizacion():
		final_ventana.set(True)
		root.destroy()


	root = Tk()
	root.config(background = "black", bd = 15)

	cat = StringVar()
	estado = BooleanVar()
	final_ventana = BooleanVar()

	 #state="disabled"
	Label(root, text="Resultado").pack()
	Entry(root, justify="center", textvariable=cat,state="disabled").pack()
	Label(root, text="").pack()
	Button(root, text="Emisor", command=emisor, height = 20, width = 20,background = "green").pack(side="left")
	Button(root, text="Emisor_Ambiental", command=emisor_ambiental, height = 20, width = 20,background = "white").pack(side="left")
	Button(root, text="Reflejo", command=reflejo, height = 20, width = 20,background = "yellow").pack(side="left")
	Button(root, text="No_Emisor", command=no_emisor, height = 20, width = 20,background = "red").pack(side="left")
	Button(root, text="Listo", command=listo, height = 20, width = 20,background = "blue").pack(side="bottom")
	Button(root, text="Omitir", command=omitir, height = 10, width = 20,background = "orange").pack(side="top")
	Button(root, text="Finalizar", command=finalizacion, height = 10, width = 20,background = "black").pack(side="top")
	root.mainloop()
	categoria = cat.get()
	est = estado.get()
	final = final_ventana.get()
	print("resultado : "+categoria)
	return categoria,est,final

def pinta_categorias(frame,categoria,x,y,w,h):
	if categoria == "Emisor":
		cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,255,0),2)
	if categoria == "Emisor_Ambiental":
		cv2.rectangle(frame ,(x,y),(x+w,y+h),(255,255,255),2)
	if categoria == "Reflejo":
		cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,255,255),2)
	if categoria == "No_Emisor":
		cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,0,255),2)

def generador_de_elpises(frame,contorno,dibujo):
	if len(contorno) >= 5:
		ellipse = cv2.fitEllipse(contorno)
		xcentroEllipse = ellipse[0][0]
		ycentroEllipse = ellipse[0][1]
		anchoEllipse = ellipse[1][0]
		altoEllipse = ellipse[1][1]
		rotationEllipse = ellipse[2]
		eccetricity = anchoEllipse/altoEllipse
		rotation = translateRotation(rotationEllipse, anchoEllipse, altoEllipse)
		if dibujo == 1:
			if (math.isnan(xcentroEllipse)): print("Nan")
			elif (math.isnan(ycentroEllipse)): print("Nan")
			elif (math.isnan(altoEllipse)): print("Nan")
			elif (math.isnan(anchoEllipse)): print("Nan")
			elif (math.isnan(rotationEllipse)): print("Nan")
			else:
				cv2.ellipse(frame, ellipse, (0,255,0), 3)
		return xcentroEllipse,ycentroEllipse,anchoEllipse,altoEllipse,rotationEllipse,rotation,eccetricity
	else: print("Contorno no cumple con el minimo de puntos")

def momentos(contorno):
	M = cv2.moments(contorno)
	print(type(M))
	return M


def caracterizador(frame,contorno,dibujo):
	caracteristicas = {}
	"""nombres = []
				numerico = []
				xcentroEllipse,ycentroEllipse,anchoEllipse,altoEllipse,rotationEllipse,rotation,eccetricity = generador_de_elpises(frame,contorno,dibujo)
				nombres = np.append(nombres,('xcentroEllipse','ycentroEllipse','anchoEllipse','altoEllipse','rotationEllipse','rotation','eccetricity'))
				parametros = np.append(numerico,(xcentroEllipse,ycentroEllipse,anchoEllipse,altoEllipse,rotationEllipse,rotation,eccetricity))
				a = 0
				for i in nombres:
					caracteristicas[i] = parametros[a]
					a+=1"""
	M = momentos(contorno)
	caracteristicas.update(M)
	return caracteristicas










"""
	#Caracterisitca 1 ============== Relación de Aspecto
	alto = recorte.shape[0]
	ancho = recorte.shape[1]
	if (alto or ancho)==0:
		alto+=1
		ancho+=1
	rel_aspecto = alto/ancho
	#Caracterisitca 2 ============== Relación lados con lados frame
	alto_frame = frame.shape[0]
	ancho_frame = frame.shape[1]
	rel_frame = (alto/alto_frame)+(ancho/ancho_frame)
	#Caracteristica 3 ============== Area
	return rel_aspecto, rel_frame"""



#Funciones copiadas de internet:
def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)
