import sys
sys.path.append('../')
import latexStrings as ls
import numpy as np
import matplotlib.pyplot as plt
from math import *
from IPython.display import Latex
import odesolver
import scipy.linalg as linear


# # Ejercicio 3

# Tomemos el siguiente PVI: 
# y_1' = - y_1 + y_2 
# y_2' =- y_1 - y_2 
# con
# y_1(0)=0 
# y_2(0)=1, en [0,1] 
# Este tiene solucion exacta:  
# y_1(t) = e^{-t}\sin(t) 
# y_2(t) = e^{-t}\cos(t) 
f = lambda t, y : np.array([-y[0]+y[1], -y[0]-y[1]])
soln = lambda t: np.array([np.exp(-T)*np.sin(T), np.exp(-T)*np.cos(T)])
y0 = np.array([0,1])

#Ahora calculemos con punto medio implicito y paso h=1/10
T,W = odesolver.solve(f, y0, (0,1), 10, method = 'Implicit Midpoint')
globalError1=linear.norm(W[:,10]-soln(T)[:,10])
dif=abs(W[:,10]-soln(T)[:,10]) #error global por entrada en t=1
print('Punto Medio Implicito h=1/10')
print('Approximaciones: '+str(W[:,10])) 
print('Diferencias Absolutas: ' + str(dif))
print('Error Global: '+str(globalError1))

#Ahora con h = 1/100
T,W = odesolver.solve(f, y0, (0,1), 100, method = 'Implicit Midpoint')
globalError2=linear.norm(W[:,100]-soln(1)[:,100])
dif=abs(W[:,100]-soln(T)[:,100]) #error global por entrada en t=1
print('Punto Medio Implicito h=1/100')
print('Approximaciones: '+str(W[:,10])) 
print('Diferencias Absolutas: ' + str(dif))
print('Error Global: '+str(globalError2))