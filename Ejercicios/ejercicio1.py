import sys
sys.path.append('../')
import latexStrings as ls
import numpy as np
import matplotlib.pyplot as plt
from math import *
from IPython.display import Latex
import odesolver


# Ejercicio 1

#Definimos nuestra funcion f, el lado derecho del PVI y la solucion exacta
f = lambda t, y : y+2*np.exp(-t)
exact = lambda t, t0, y0 : np.exp(t)*(y0*np.exp(-t0)+np.exp(-2*t0)-np.exp(-2*t))

#Guardamos y0 asi como los extremos de nuestro intervalo I
I = (0, 1)
y0 = 1

#Calculamos el valor exacto al tiempo 1
y1 = exact(1,0,1)

#Hacemos Euler Explicito con 10 pasos
T, W = odesolver.solve(f,y0,I,10,method="Explicit Euler")

#Calculamos errores globales y locales e imprimimos resultados
globalError = abs(W[0,10]-y1)
localErrors = [abs(W[0,i+1] - exact(T[i+1], T[i], W[0,i])) for i in range(10)]
maxLocalError = max(localErrors)

print("Aproximaciones: ")
print(W[0])
print("Error Global: "+str(globalError))
print("Maximo Error Local: "+str(maxLocalError))


#Grafica de la solucion
plt.plot(T,exact(T,0,1))
plt.title("Euler Explicito")
plt.plot(T,W[0], c='r')
plt.scatter(T,W[0], c='r')
plt.show()


#Calculamos Euler explicito con pasos h = 0.1**2^-k , k=0,1,...,5
res = []
globalErrors=[]

for k in range(6):
    h = 0.1*(2**-k)
    m = int((I[1]-I[0])/h)
    
    T, W = odesolver.solve(f, y0, I, m, method="Explicit Euler")
    
    g = abs(W[0,m]-y1)
    globalErrors.append(g)
    
    localErrors = [abs(W[0,i+1] - exact(T[i+1], T[i], W[0,i])) for i in range(m)]
    maxLocalError = max(localErrors)
    
    eoc = "NaN" if k==0 else log(g/prevg)/log(h/prevh)
    res.append([k, h, maxLocalError, eoc])
    
    prevh = h
    prevg = g


#Observemos el comportamiento del error del metodo en relacion con el tamaño del paso
header = ["k", "Paso", "Error Local Maximo", "eoc"]
print(ls.latexTable(header, res, '|c|r r r|'))

#Grafica del error global contra pasos
steps = [i[1] for i in res]
plt.title("Error Global de Euler Explicito")
plt.plot(steps,globalErrors)
plt.scatter(steps, globalErrors)
plt.xscale('log')
plt.xlabel('Paso (log)')
plt.yscale('log')
plt.ylabel('Error Global (log)')
plt.axis([1e-3,1,1e-3,1])
plt.show()
