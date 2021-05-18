
import cv2
import numpy as np

import pandas as pd

caracteristicas = pd.DataFrame()
diccionario = {"Hola":1,"Hola2":2}
diccionario["Hola3"] = 3
caracteristicas = caracteristicas.append(diccionario, ignore_index=True)
caracteristicas = caracteristicas.append({"Hola2":2}, ignore_index=True)

print(caracteristicas)