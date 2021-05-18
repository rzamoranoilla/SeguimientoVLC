import cv2

cams_test = 20
for i in range(-1, cams_test):
	cap = cv2.VideoCapture(i)
	test, frame = cap.read()
	if test == True:
		print("i : "+str(i)+" /// result: "+str(test))
	else: print(str(i)+" listo")