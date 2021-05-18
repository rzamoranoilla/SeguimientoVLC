import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib.transforms as mtransforms

frame = cv2.imread("C:/Users/raul/Desktop/8.- Tesis/5.- Redes Neuronales/Muestras/Muestras/COCO/emisores/000000000001.jpg")
#frame = cv2.imread("./muestras/imagenes/emisores/000000000001.jpg")

I = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
hist = I.flatten()
#cv2.imshow("Hist",hist)

#plt.axvline(x=255*0.85, ymin=0, ymax=1,linestyle = '--')
#plt.axvline(x=255, ymin=0, ymax=1,linestyle = '--')


fig, ax = plt.subplots()

t = np.linspace(0,255,len(hist))
y = np.linspace(0,200000,len(hist))
orden = np.sort(hist)
print(int(len(hist)*0.90))
print(orden[int(len(hist)*0.90)])
#plt.plot(t,y)
#y1 = [0,0,250]
#y2 = 260

#plt.fill_between(t,0,200)

plt.hist(hist,bins=int(len(hist)/100),histtype="step",color="red")
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.fill_between(t, 0, 1, where=y > (200000/255)*orden[int(len(hist)*0.90)],
                facecolor='lightblue', alpha=0.5, transform=trans)
ax.fill_between(t, 0, 1, where=y > (200000/255)*orden[int(len(hist)*0.90)],
                facecolor='lightgreen', alpha=0.5, transform=trans)
plt.xlabel("Valor de Intensidad Luminosa (Int8)")
plt.ylabel("Cantidad de Pixels")
plt.show()

cv2.imshow("Frame",frame)
while(True):
	if cv2.waitKey(1) & 0xFF == ord('s'):
		break

