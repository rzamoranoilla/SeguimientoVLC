import pandas as pd
import numpy as np
from nn_utils import NetworkDefinition, ForwardPropagation, BackwardPropagation, ParametersUpdate, LossFunction
from nn_utils import loadData, Accuracy, PlotLoss, PlotDecisionLines

df = pd.read_csv("datos_entrenamiento.csv")

def traductor(df,n_categorias):
	#Toma los valores de entrada del dataframe y los ordena en dos listas:
	# - Categorias (Son las categorias que se)
	#
	categorias = []
	caracteristicas = []
	for i in df.values.tolist():
		categoria = []
		caracteristica = []
		a = 1
		while(a<=n_categorias):
			categoria.append(i[a])
			a+=1
		while(a<=len(df.values.tolist())):
			caracteristica.append(i[a])
			a+=1
		categorias.append(categoria)
		caracteristicas.append(caracteristica)
	#print(categorias)
	#print(caracteristicas)
	return categorias,caracteristicas

def potencia(c):
    """Calcula y devuelve el conjunto potencia del 
       conjunto c.
    """
    if len(c) == 0:
        return [[]]
    r = potencia(c[:-1])
    return r + [s + [c[-1]] for s in r]

def combinador(c, n):
    """Calcula y devuelve una lista con todas las
       combinaciones posibles que se pueden hacer
       con los elementos contenidos en c tomando n
       elementos a la vez.
    """
    return [s for s in potencia(c) if len(s) == n]

def comb_caract(df,n_categorias,n_caracteristicas_entrenamiento):
	#Entradas:
	#df = Dataframe
	#n_categorias = Numero de categorias
	#n_categorias_entrenamiento = Numero de categorias que se ocupará para entrenar cada modelo de red

	categorias1,caracteristicas1 = traductor(df,n_categorias)
	#print(categotias1)
	#print(caracteristicas1)
	#print("============================================================================")
	comb_caracteristicas = {}
	b = 0
	for i in caracteristicas1:
		combinaciones = combinador(i,n_caracteristicas_entrenamiento)
		a =0
		while(a<len(combinaciones)):
			if b == 0:
				comb_caracteristicas["COMB"+str(a)] = [combinaciones[a]]
				#b = 1
			else:
				comb_caracteristicas["COMB"+str(a)].append(combinaciones[a])
				#print(pares_caracteristicas)
			a+=1
		b = 1
		#while(a<len(combinaciones)):
	#print(comb_caracteristicas["par1"])
	return categorias1,comb_caracteristicas


def entrenar():
	categorias, combinaciones = comb_caract(df,3,2)
	cat_numpy = np.array(categorias)
	print(cat_numpy)
	print(type(cat_numpy))
	#print(combinaciones)
	for i in combinaciones:
		print(i)
		#print(combinaciones[i])
		comb = np.array(combinaciones[i])
		print(comb)
		print(len(combinaciones[i]))
		X = comb
		Y = cat_numpy

		# Loading Data
		test_size    = 0.20                               # 80% train 20% test
		(Xtrain, Ytrain, Xtest, Ytest) = loadData(X,Y,1,test_size,1)
		N           = Xtrain.shape[1]                     # training samples
		n_0         = Xtrain.shape[0]                     # number of inputs (X)
		n_m         = Ytrain.shape[0]                     # number of outputs (Y)

		# Definitions
		tmax        = 1000                                # max number of iterations
		alpha       = 10                                  # learning rate
		loss_eps    = 0.01                                # stop if loss<loss_eps
		nh          = [6,12]                              # nodes of hidden layers
		n           = [n_0]+nh+[n_m]                      # nodes of each layer
		m           = len(n)-1
		ltrain      = np.zeros([tmax,1])                  # training loss
		W,b         = NetworkDefinition(n,N)              # (step 1)

		# Training
		t  = -1
		ok = 0
		training = 1
		while training:
		    t         = t+1
		    a         = ForwardPropagation(Xtrain,W,b)    # (step 2)
		    dW,db     = BackwardPropagation(Ytrain,a,W,b) # (step 3)
		    W,b       = ParametersUpdate(W,b,dW,db,alpha) # (step 4)
		    ltrain[t] = LossFunction(a,Ytrain)            # (step 5)
		    training  = ltrain[t]>=loss_eps and t<tmax-1

		# Evaluation
		a   = ForwardPropagation(Xtrain,W,b)    # output layer is a[m]
		acc = Accuracy(a[m],Ytrain,'Training')

		a   = ForwardPropagation(Xtest,W,b)     # output layer is a[m]
		acc = Accuracy(a[m],Ytest,'Testing')

		# Plots
		PlotLoss(ltrain)
		PlotDecisionLines([0,1],[0,1],W,b,a)



entrenar()

