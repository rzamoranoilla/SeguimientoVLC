from tkinter import *

def emisor():
	cat.set('Emisor')
	print(cat)

def no_emisor():
	cat.set('Emisor')
	print(cat)

# Estructura del formulario
root = Tk()
root.config(bd=90)  # borde exterior de 15 píxeles, queda mejor

cat = StringVar()

Label(root).pack() # Separador
Button(root, text="Emisor", command = emisor).pack()
Label(root).pack() # Separador
Button(root, text="No Emisor", command = no_emisor).pack()
root.mainloop()