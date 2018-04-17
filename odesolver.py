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

def EEStep(f, wi, ti, h):
    w = wi + h*f(ti, wi)
    
    return w
    
def IEStep(f, wi, ti, h):
    F = lambda x : x - wi- h*f(ti, x) 
    #Uses explicit euler as an initial guess
    guess  = wi + h*f(ti , wi)
    w = op.newton_krylov(F, guess)
    
    return w

def ETStep(f, wi, ti, h):
    S0 = f(ti , wi)
    S1 = f(ti + h, wi + h*S0)
    w = wi + h/2*(S0+S1)
    
    return w

def ITStep(f, wi, ti, h):
    S0 = f(ti , wi)
    F = lambda x : x - wi- h/2*(S0 + f(ti+h, x))
    #Uses explicit euler as an initial guess
    guess  = wi + h*S0
    w = op.newton_krylov(F, guess)
    
    return w

def EMPStep(f, wi, ti, h):
    S0 = f(ti , wi)
    S1 = f(ti + h/2, wi+h/2*S0)
    w = wi + h*S1
    
    return w
    
def RK4(f, y0, I, m):
    h, T, W = initODE(y0, I, m)
    
    for i in range(m):
        S0 = f(T[i] , W[:,i])
        S1 = f(T[i] + h/2, W[:,i] + h/2*S0)
        S2 = f(T[i] + h/2, W[:,i] + h/2*S1)
        S3 = f(T[i] + h, W[:,i] + h*S2)
        W[:,i+1] = W[:,i] + h/6*(S0+2*S1+2*S2+S3)
    
    return T, W
    
def RK4Step(f, wi, ti, h):
    S0 = f(ti , wi)
    S1 = f(ti + h/2, wi + h/2*S0)
    S2 = f(ti + h/2, wi + h/2*S1)
    S3 = f(ti + h, wi + h*S2)
    w = wi + h/6*(S0+2*S1+2*S2+S3)
    
    return w

def RK23Step(f, wj, tj, hj, tol = 1e-4, maxStep=np.inf):
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
        
        return [w, tj+hj, min(2*hj,maxStep)]
    else:
        return RK23Step(f, wj, tj, hj/2, tol, maxStep)

def solve(f, y0, I, m = 1e3, method = 'RK4', tol = 1e-4, maxStep = np.inf):
    h, T, W = initODE(y0, I, m)
    method = method.casefold()
    if method == 'explicit euler':
        
        for i in range(m):
            W[:,i+1]=EEStep(f, W[:,i], T[i], h)
        
    elif method == 'implicit euler':
        
        for i in range(m):
            W[:,i+1]=IEStep(f, W[:,i], T[i], h)
        
    elif method == 'explicit trapezoid':
        
        for i in range(m):
            W[:,i+1]=ETStep(f, W[:,i], T[i], h)
        
    elif method == 'implicit trapezoid':
        
        for i in range(m):
            W[:,i+1]=ITStep(f, W[:,i], T[i], h)
        
    elif method == 'explicit midpoint':
        
        for i in range(m):
            W[:,i+1]=EMPStep(f, W[:,i], T[i], h)
        
    elif method == 'rk4':
        
        for i in range(m):
            W[:,i+1]=RK4Step(f, W[:,i], T[i], h)
        
    elif method == 'rk23':
        t = I[0]
        w = y0
        
        T = [t]
        W = [w]
        
        while t<I[1]:
            w, t, h = RK23Step(f, w, t, h, tol, maxStep)
            T.append(t)
            W.append(w)
            
            if t+h>I[1]:
                h = I[1]-t
            
        T = np.array(T) 
        W = np.array(W)
    else:
        raise ValueError('Unsupported method')
    return T,W