import sys
sys.path.append('../')
import latexStrings as ls
import numpy as np
import matplotlib.pyplot as plt
from math import *
from IPython.display import Latex
import odesolver


# # Ejercicio 4 


#funcion
f = lambda t, y : 5*y-3*y*y

#Diagrama de flujo
T = np.linspace(-1,1,25)
Y = np.linspace(-1,3,25)

F = np.array([[f(t,y) for t in T] for y in Y])
F = np.arctan(F)
U = np.cos(F)
V = np.sin(F)

plt.quiver(T,Y,U,V)
plt.show()

# Ahora apliquemos el metodo de Euler implicito con paso $h=0.5$ a nuestra función
I = (0, 20)
y0 = 0.5
T, W = odesolver.solve(f,y0,I,40,method="Implicit Euler")
print('y(20) con EI, 40 pasos: ' + str(W[0,40]))

#EE con 40,80,100 y 120 pasos
T2, W2 = odesolver.solve(f,y0,I,120,method="Explicit euler")
T3, W3 = odesolver.solve(f,y0,I,100,method="Explicit Euler")
T4, W4 = odesolver.solve(f,y0,I,80,method="Explicit Euler")
T5, W5 = odesolver.solve(f,y0,I,40,method="Explicit Euler")

print('y(20) con EE, 40 pasos: ' + str(W5[0,40]))
print('y(20) con EE, 80 pasos: ' + str(W4[0,80]))
print('y(20) con EE, 100 pasos: ' + str(W3[0,100]))
print('y(20) con EE, 120 pasos: ' + str(W2[0,120]))

#RK2/3 con tol 0.1 y 0.01
T6, W6 = odesolver.solve(f,y0,I,method="RK23",tol=0.1,initialStep=0.5)
print('y(20) con RK2/3, tol = 0.1:' + str(W6[0,-1]))
print('Total de aproximaciones: ' + str(len(W6[0])-1))


T7, W7 = odesolver.solve(f,y0,I,method="RK23",tol=0.01,initialStep=0.5)
print('y(20) con RK2/3, tol = 0.01:' + str(W7[0,-1]))
print('Total de aproximaciones: ' + str(len(W7[0])-1))

