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
# \begin{equation} \label{eq:3}
#  \begin{array}{c} y_1' = - y_1 + y_2 \\
# y_2' =- y_1 - y_2 \end{array} 
# \qquad \mbox{con} \qquad \begin{array}{c}
# y_1(0)=0 \\ y_2(0)=1 \end{array} \qquad \mbox{para } t\in [0,1] \qquad 
# \end{equation}
# 

# con solución exacta  $\overrightarrow{y}(t)=\left(\begin{array}{c} y_1(t) \\
# y_2(t)\end{array} \right)$ está dada por :
# \begin{equation} \label{eq:4}
#  \begin{array}{c} y_1(t) = e^{-t}\sin(t) \\
# y_2(t) = e^{-t}\cos(t) \end{array}
# \end{equation}

# Sea $\overrightarrow{W}_j$ el vector de aprximación correspondiente a la aplicación del método del Punto Medio Implícito con paso $h_j$. La entrada $w_i$ es la aproximación en $t=1$ de $y_i$ para $i,j \in \{1,2\}$

# Sea $E_{PMImpl}(h_j)=\Vert W_j-\overrightarrow{y}(1)\Vert _2$ el error global en $t=1$ para $j \in \{1,2\}$ 
# 

# $\overrightarrow{G}_j:=\vert W_j-\overrightarrow{y}(1) \vert$ $\quad$cuya entrada $g_i=\vert w_1-y_i(1) \vert$ 

# Aplicaremos el método  para $h_1=0.1$ y $h_2=0.01$:

# In[2]:


f = lambda t, y : np.array([-y[0]+y[1], -y[0]-y[1]])
soln = lambda t: np.array([np.exp(-T)*np.sin(T), np.exp(-T)*np.cos(T)])
y0 = np.array([0,1])
T,W = odesolver.solve(f, y0, (0,1), 10, method = 'Implicit Midpoint')
globalError1=linear.norm(W[:,10]-soln(T)[:,10])
dif=abs(W[:,10]-soln(T)[:,10]) #error global por entrada en t=1
Latex('$h_1=0.1$'+ ls.latexVector(W[:,10],'W_1',form='%f')+ ls.latexVector(dif,'G_1',form='%f') +' $E_{PMImpl}(h_1) = '+str(globalError1)+'$')


# In[3]:


T,W = odesolver.solve(f, y0, (0,1), 100, method = 'Implicit Midpoint')
globalError2=linear.norm(W[:,100]-soln(1)[:,100])
dif=abs(W[:,100]-soln(T)[:,100]) #error global por entrada en t=1
Latex('$h_2=0.01$'+ ls.latexVector(W[:,100],'W_2',form='%f')+ ls.latexVector(dif,'G_2',form='%f') + '$E_{PMImpl}(h_2) = '+str(globalError2)+'$')


# El método del Punto Medio Implícito con paso $  \hat{h}  $ tiene un error global de de orden 2 la forma: $$E_{PMImpl}=0\left (\hat{h}^2\right)$$ 

# Con $\hat{h}:=h_1=0.1$ obtenemos: $$E_{PMImpl}(h_1)=O\left ( h_1^2 \right ) = O \left( 0.1^2 \right)$$

# Con $\hat{h}:=h_2=0.01$ obtenemos: $$E_{PMImpl}(h_2)=O\left ( h_2^2 \right ) = O \left( 0.01^2 \right)$$

# Observamos lo siguiente: $$E_{PMImpl}(h_1)=O\left ( h_1^2 \right ) $$ $$  E_{PMImpl}(h_2)=O(h_2^2)$$
# $$  h_2= \frac{h_1}{10}   \Rightarrow      E_{PMImpl}(h_2) =O\left(\frac{h_1}{10}\right)^2  $$
# $$ \Rightarrow E_{PMImpl}(h_2) \approx \frac{1}{100}O(h_1^2)=\frac{1}{100}E_{PMImpl}(h_1) $$

# Esto implica que el error global del método para $h_2$ es aproximadamente 100 veces menor al error global con el paso $h_1$.
# 

# In[4]:


Latex('$'+'(0.01)' + 'E_{PMImpl}(h_1)='+str(globalError1/100) + '$')


# In[5]:


Latex('$ E_{PMImpl}(h_2)='+str(globalError2) + '$')  


# En la operación anterior podemos observar que la reducción del error de $h_1$ a $h_2=\frac{h_1}{10}$es consistente  ya que: $$ E_{PMImpl}(h_2)=8.671073839286167e−06 \approx 8.67819813142771e−06= \frac{1}{100}E_{PMImpl}(h_1) $$
