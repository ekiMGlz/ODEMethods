import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from math import *
import odesolver
from scipy import optimize as op

# # Ejercicio 6

print('\nShooting Method\n')

print('Tomemos el siguiente PVI:')

print("y'' = 10y' (1 - y)")
print("y(0) = 1")
print('y(1) = 1 - pi/20}')
print('Para t \in [0,1]')

print('\nQue podemos reescribir como un sistema de ecuaciones de primer orden de la siguiente manera:')
print("y_1' = y_2")
print("y_2' = 10 y_2 (1 - y_1)")
print('\ncon solución exacta dada por :')
print("y(t) = 1-pi/20*tan(pi/4*t)")
print('\nasí, definimos f como')
print("f = lambda t, y : np.array([y[1], 10*y[1]*(1-y[0])])")
f = lambda t, y : np.array([y[1], 10*y[1]*(1-y[0])])
print('\ny la solución exacta como')
print('exact = lambda t : 1-pi/20*tan(pi/4*t)')
exact = lambda t : 1-pi/20*tan(pi/4*t)

print("\nNotamos que")
print('exact(1) == 1-pi/20: '+ str(exact(1) == 1-pi/20))

print('\n\nPara aplicar Shooting Method, tomamos una aproximación inicial de y_2 = 0')

print('y_1(t) = 1')
print('y_2(t) = 0')

print("\ny utilizaremos el método de Runge Kutta 4 con 1000 pasos para aproximar la función en y_1(1)")

y0 = [1,0]

_, W = odesolver.solve(f, y0, (0,1), 1000,  method = 'rk4')


# In[7]:

ans = W[0][-1]
print('W(1) = '+ str(ans))


print('\nVeamos que tan acertada fue nuestra elección')

print('abs(W(1) - (1 - pi/20)) = ' +str(abs(ans - (1 - pi/20))))

print('\nParece ser que nuestra aproximación inicial sobreestimó la solución. Usaremos el método de Newton para seguir generando aproximaciones.')

print("\n\nDefinimos una función F : R -> R que toma una estimación inical de y_2 = y' y regresa la distancia a la solución delimitada por el BVP")

# In[9]:

I = (0,1)
y0 = 1
sol = 1 - pi/20
print("F = lambda yp0 : odesolver.solve(f, [y0, yp0], I, 1000,  method = 'rk4')[1][0][-1] - sol")
F = lambda yp0 : odesolver.solve(f, [y0, yp0], I, 1000,  method = 'rk4')[1][0][-1] - sol


# In[10]:

print('F(0) = ' + str(F(0)))

print('En odesolver, definimos la función "shooting" que toma una función, un valor inicial de y_1, una aproximación inicial de y_2 y la la solución del BVP y crea la función F, para después aplicar el método de Newton hasta encontrar una raíz, o bien, encontrar el valor inicial de $y_2$ que hace que se satisfaga el BVP.\n')

aprox, errors = odesolver.shooting(f, 1, 0, exact(1), (0,1), tol = 1e-10, maxiter=30, log = True)

print('\nObservamos que con esta última aprximación inicial se satisface el BVP con un error menor a 1e-10')

print("\nAproximaciones de y' inicial generadas por el método de Newton:")
print(aprox)

plt.scatter(range(7), [0]+aprox)
plt.title("Aproximaciones de y' inicial generadas por el método de Newton")
plt.xlabel('Iteracion')
plt.ylabel('y\'')
plt.show()

# In[15]:

print("Error generado por cada aproximación, usando RK4 con 1000 pasos:")
print(errors)


plt.scatter(range(7), [F(0)]+errors)
plt.title("Error generado por cada aproximación, usando RK4 con 1000 pasos")
plt.xlabel('Iteracion')
plt.ylabel('Error')
plt.show()




