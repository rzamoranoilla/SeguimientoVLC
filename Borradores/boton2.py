from tkinter import *

def funcion_pronter():
	def emisor():
	    cat.set( "Emisor" )

	def no_emisor():
	    cat.set(" No_Emisor" )

	def listo():
		estado.set(True)
		print("a")
		root.destroy()

	root = Tk()
	root.config(bd=15)

	cat = StringVar()
	estado = BooleanVar()

	 #state="disabled"
	Label(root, text="Resultado").pack()
	Entry(root, justify="center", textvariable=cat).pack()
	Label(root, text="").pack()
	Button(root, text="Emisor", command=emisor).pack(side="left")
	Button(root, text="No_Emisor", command=no_emisor).pack(side="left")
	Button(root, text="Listo", command=listo).pack(side="left")
	if estado == True:
		print("a")


	root.mainloop()
	categoria = cat.get()
	print("resultado : "+categoria)

funcion_pronter()