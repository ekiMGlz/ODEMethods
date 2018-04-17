import numpy as np
import scipy.optimize as op
from math import *

def initODE(y0, I, m):
    h = (I[1]- I[0])/m
    T = np.linspace(I[0],I[1],m+1)
    
    try:
        n = len(y0)
    except Exception:
        n = 1
    
    W = np.zeros((n, m+1))
    W[:,0] = y0
    
    return h, T, W

def explicitEuler(f, y0, I, m):
    
    h, T, W = initODE(y0, I, m)
    
    for i in range(m):
        W[:,i+1] = W[:,i] + h*f(T[i] , W[:,i])
    
    return T, W

def implicitEuler(f, y0, I, m):
    
    h, T, W = initODE(y0, I, m)
    
    for i in range(m):
        F = lambda w : w - W[:,i]- h*f(T[i+1], w) 
        #Uses explicit euler as an initial guess
        guess  = W[:,i] + h*f(T[i] , W[:,i])
        W[:,i+1] = op.newton_krylov(F, guess)
    
    return T, W

def explicitTrap(f, y0, I, m):
    h, T, W = initODE(y0, I, m)
    
    for i in range(m):
        S0 = f(T[i] , W[:,i])
        S1 = f(T[i] + h, W[:,i] + h*S0)
        W[:,i+1] = W[:,i] + h/2*(S0+S1)
    
    return T, W

def implicitTrap(f, y0, I, m):
    h, T, W = initODE(y0, I, m)
    
    for i in range(m):
        S0 = f(T[i] , W[:,i])
        F = lambda w : w - W[:,i]- h/2*(S0 + f(T[i+1], w))
        #Uses explicit euler as an initial guess
        guess  = W[:,i] + h*S0
        W[:,i+1] = op.newton_krylov(F, guess)
    
    return T, W

def explicitMP(f, y0, I, m):
    h, T, W = initODE(y0, I, m)
    
    for i in range(m):
        S0 = f(T[i] , W[:,i])
        S1 = f(T[i] + h/2, W[:,i]+h/2*S0)
        W[:,i+1] = W[:,i] + h*S1
    
    return T, W

def RK4(f, y0, I, m):
    h, T, W = initODE(y0, I, m)
    
    for i in range(m):
        S0 = f(T[i] , W[:,i])
        S1 = f(T[i] + h/2, W[:,i] + h/2*S0)
        S2 = f(T[i] + h/2, W[:,i] + h/2*S1)
        S3 = f(T[i] + h, W[:,i] + h*S2)
        W[:,i+1] = W[:,i] + h/6*(S0+2*S1+2*S2+S3)
    
    return T, W

def RK23step(f, tj, wj, hj, tol = 1e-4, maxStep=np.inf):
    """
    IN
        f - function
        tj - Start time
        wj - Last aproximation
        hj - suggested step
        tol - tolerance
        maxStep - maximum step size
    OUT
       tn - End time
       wn - approximation
       hn - next suggested step
    """
    
    S1 = f(tj,wj)
    S2 = f(tj,wj+hj*S1)
    S3 = f(tj,wj + hj*(S1+S2)/4)
    e = hj*abs(S1-2*S3+S2)/3
    
    if e < tol*max(abs(wj), 1):
        
        w = wj+hj/6*(S1+4*S3+S2)
        
        return [tj+hj, w, min(2*hj,maxStep)]
    else:
        return RK23step(f, tj, wj, hj/2, tol, maxStep)

def RK23(f, I, y0, h = 1, tol = 1e-4, maxStep=np.inf):
    """
    IN:
        f - function
        I - Interval
        y0 - initial value
        h - suggested initial step
        tol - tolerance
        maxStep - maximum step length
    OUT:
        T - times
        W - values
    """
    t = I[0]
    w = y0
    
    T = [t]
    W = [w]
    
    while t<I[1]:
        t,w,h = RK23step(f, t, w, h, tol, maxStep)
        T.append(t)
        W.append(w)
        
        if t+h>I[1]:
            h = I[1]-t
        
    return np.array(T), np.array(W)