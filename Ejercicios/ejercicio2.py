import sys
sys.path.append('../')
import latexStrings as ls
import numpy as np
import matplotlib.pyplot as plt
from math import *
from IPython.display import Latex
import odesolver


# # Ejercicio 2

#Tomando el PVI del ejercicio 1
f = lambda t, y : y+2*np.exp(-t)
exact = lambda t, t0, y0 : np.exp(t)*(y0*np.exp(-t0)+np.exp(-2*t0)-np.exp(-2*t))
I = (0, 1)
y0 = 1
y1 = exact(1,0,1)

#Resolvemos con trapecio explicito y pasos h=1/10
T, W = odesolver.solve(f,y0,I,10,method="Explicit Trapezoid")
globalError = abs(W[0,10]-y1)
print("Error Trapecio Explicito h = 1/10: "+str(globalError))


#Veamos lo que sucede cuando usamos el paso h/2=0.05
T, W = odesolver.solve(f,y0,I,20,method="Explicit Trapezoid")
globalError = abs(W[0,20]-y1)
print("Error Trapecio Explicito h = 1/20: "+str(globalError))


#Ahora calcuemos el error global en t=1 usando el método Runge-Kutta 4 usando h1: 
T, W = odesolver.solve(f,y0,I,10,method="RK4")
globalError = abs(W[0,10]-y1)
print("Error RK4 h = 1/10: "+str(globalError))