import sys
sys.path.append('../')
import latexStrings as ls
import numpy as np
import matplotlib.pyplot as plt
from math import *
from IPython.display import Latex
import odesolver


# # Ejercicio 2

# Tomando el PVI \ref{eq:1} y la solución general de la ecuación \ref{eq:2} del ejercicio anterior aplicaremos el método del Trapecio explícito con paso $h_1=0.1$. En el ejercicio 1 se argumentó la existencia y unicidad del PVI. 
# 
# Calculemos el error global en el punto t = 1: $$E_{Trapecio}(h_1)=\left|w_{10}-y(1) \right|$$ 
# con $w_{10}$ la aproximación del método en t=1. 

# In[2]:


f = lambda t, y : y+2*np.exp(-t)
exact = lambda t, t0, y0 : np.exp(t)*(y0*np.exp(-t0)+np.exp(-2*t0)-np.exp(-2*t))
I = (0, 1)
y0 = 1

y1 = exact(1,0,1)

T, W = odesolver.solve(f,y0,I,10,method="Explicit Trapezoid")

globalError = abs(W[0,10]-y1)

Latex("$E_{Trapecio}(h_1)= "+str(globalError) +'$' )


# Veamos lo que sucede cuando usamos el paso $h_2:=\frac{h_1}{2}=0.05$ y calculemos el correspondiende error global en t=1: $$E_{Trapecio}(h_2)=\left|w_{20}-y(1) \right|$$ 

# In[3]:


T, W = odesolver.solve(f,y0,I,20,method="Explicit Trapezoid")

globalError = abs(W[0,20]-y1)

Latex("$E_{Trapecio}(h_2)= "+str(globalError) +'$')


# Ahora calcuemos el error global en t=1 usando el método Runge-Kutta 4 usando $h_1=0.1$ :
# $$E_{RK4}(h_1)=\left|w_{10}-y(1) \right|$$ 
# 

# In[4]:


T, W = odesolver.solve(f,y0,I,10,method="RK4")

globalError = abs(W[0,10]-y1)

Latex("$E_{RK4}(h_1)= "+str(globalError) +'$')


# \begin{table}[h]
# \centering
# \begin{tabular}{|l|lll|}
# \hline
#                                    & E$_{Trapecio}(h_1)$   & E$_{Trapecio}(h_2)$   & E$_{RK4}(h_1)$                       \\ \hline
# Error (global en t = 1)            & 4.63749554010 $\times$ 10$^{-3}$& 1.22097781614 $\times$ 10$^{-3}$& 2.3348121258 $\times$ 10$^{-6}$ \\
# n\'umero de estados & 20                    & 40                    & 40                                  \\ \hline
# \end{tabular}
# \end{table}

# En la tabla anterior notamos lo siguiente: $$E_{RK4}(h_1)<E_{Trapecio}(h_2)<E_{Trapecio}(h_1)$$

# Podemos observar que mediante el método del Trapecio explícito con paso $h_2$ obtenemos un error aproximadamente 4 veces menor que con paso $h_1$ ya que: $$E_{Trapecio}(h_1)=O(h_1^2)$$ $$   \Rightarrow E_{Trapecio}(h_2)=O(h_2^2)=O\left(\frac{h_1}{2}\right)^2$$ $$ \Rightarrow E_{Trapecio}(h_2) \approx \frac{1}{4}O(h_1^2)=\frac{1}{4}E_{Trapecio}(h_1) $$
# El problema es que el método con $h_2$ necesita de 40 estados, el doble con respecto a usar $h_1$. 

# Por otro lado, el método Runge-Kutta 4 necesita de 40 estados al igual que el Trapecio explícito con $h_2$ pero observando los resultados de la tabla notamos que el error global $E_{RK4}$ es  menor a $E_{Trapecio}(h_2)$ por un factor de $10^3$ aproximadamente.

# Por lo tanto, para reducir el error, es preferible usar el método Runge-Kutta 4 ya que, por los mismos 40 estados requeridos, se obtiene un error menor al del Trapecio explícito con $h_2$.
