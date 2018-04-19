import numpy as np
import scipy.optimize as op
from math import *

def initODE(y0, I, m):
    """
    Initializes T, the vales of t for the approximation, and W, the approximations, for an IVP
    
    Parameters
    ----------
    y0 : (N) double ndarray
        The initial values for y.
    I : tuple (t0 tf)
        The interval (t0, tf) over which to approximate the solution.
    m : int, optional
        Number of steps in the approximation.
    
    Returns
    -------
    h : double
        The size of the steps taken between approximations.
    T : (m+1) double ndarray
        A regularly spaced vector of times starting at t0 and ending at t1 with steps h.
    W : (N, m+1) double ndarray
        A matrix of zeros with y0 in the first column.
    """
    
    #Define h so that the m steps taken are regularly spaced, then save the linespace in T
    h = (I[1]- I[0])/m
    T = np.linspace(I[0],I[1],m+1)
    
    #Seve the dimmension of y0 as n, throws exception when y0 is a number, in which case n = 1
    try:
        n = len(y0)
    except Exception:
        n = 1
    
    #Initialize W as a n x (m+1) zero matrix, and save y0 in the first column
    W = np.zeros((n, m+1))
    W[:,0] = y0
    
    return h, T, W

def EEStep(f, wi, ti, h):
    """
    Explicit Euler step for approximating ordinary differential equations.
    The step is given by::
        w = wi + h*f(ti, wi)
    This method has a local error of O(h^2), and a global error of O(h).
    
    Parameters
    ----------
    f : function
        Right hand side of the ordinary differential equation.
    y0 : (N) double ndarray
        The initial values for y.
    h : double
        The step taken.
    
    Returns
    -------
    w : (N) double ndarray
        The approximation at ti+h
    """
    
    w = wi + h*f(ti, wi)
    
    return w
    
def IEStep(f, wi, ti, h):
    """
    Implicit Euler step for approximating ordinary differential equations.
    The step is given by::
        w = wi + h*f(ti, w)
    This method has a local error of O(h^2), and a global error of O(h).
    In order to solve for w, the method fsolve from scipy.optimize is used, 
    using the Explicit Euler approximation as an initial guess.
    
    Parameters
    ----------
    f : function
        Right hand side of the ordinary differential equation.
    y0 : (N) double ndarray
        The initial values for y.
    h : double
        The step taken.
    
    Returns
    -------
    w : (N) double ndarray
        The approximation at ti+h
    """
    
    F = lambda x : x - wi- h*f(ti, x) 
    guess  = wi + h*f(ti , wi)
    w = op.fsolve(F, guess)
    
    return w

def ETStep(f, wi, ti, h):
    """
    Explicit Trapezoid step for approximating ordinary differential equations.
    The step is given by::
        w = wi + h/2*(f(ti, wi) + f(ti + h, wi + h*f(ti, wi)))
    This method has a local error of O(h^3), and a global error of O(h^2).
    
    Parameters
    ----------
    f : function
        Right hand side of the ordinary differential equation.
    y0 : (N) double ndarray
        The initial values for y.
    h : double
        The step taken.
    
    Returns
    -------
    w : (N) double ndarray
        The approximation at ti+h
    """
    
    S0 = f(ti , wi)
    S1 = f(ti + h, wi + h*S0)
    w = wi + h/2*(S0+S1)
    
    return w

def ITStep(f, wi, ti, h):
    """
    Implicit Trapezoid step for approximating ordinary differential equations.
    The step is given by::
        w = wi + h/2*(f(ti, wi) + f(ti, w))
    This method has a local error of O(h^3), and a global error of O(h^2).
    In order to solve for w, the method fsolve from scipy.optimize is used, 
    using the Explicit Euler approximation as an initial guess.
    
    Parameters
    ----------
    f : function
        Right hand side of the ordinary differential equation.
    y0 : (N) double ndarray
        The initial values for y.
    h : double
        The step taken.
    
    Returns
    -------
    w : (N) double ndarray
        The approximation at ti+h
    """
    
    S0 = f(ti , wi)
    
    F = lambda x : x - wi- h/2*(S0 + f(ti+h, x))
    guess  = wi + h*S0
    w = op.fsolve(F, guess)
    
    return w

def EMPStep(f, wi, ti, h):
    """
    Explicit Midpoint step for approximating ordinary differential equations.
    The step is given by::
        w = wi + h*f(ti + h/2, wi + h/2*f(ti, wi))
    This method has a local error of O(h^3), and a global error of O(h^2).
    
    Parameters
    ----------
    f : function
        Right hand side of the ordinary differential equation.
    y0 : (N) double ndarray
        The initial values for y.
    h : double
        The step taken.
    
    Returns
    -------
    w : (N) double ndarray
        The approximation at ti+h
    """
    
    S0 = f(ti , wi)
    S1 = f(ti + h/2, wi+h/2*S0)
    w = wi + h*S1
    
    return w

def IMPStep(f, wi, ti, h):
    """
    Implicit Midpoint step for approximating ordinary differential equations.
    The step is given by::
        w = wi + h*f(ti + h/2, (wi+w)/2)
    This method has a local error of O(h^3), and a global error of O(h^2).
    In order to solve for w, the method fsolve from scipy.optimize is used, 
    using the Explicit Euler approximation as an initial guess.
    
    Parameters
    ----------
    f : function
        Right hand side of the ordinary differential equation.
    y0 : (N) double ndarray
        The initial values for y.
    h : double
        The step taken.
    
    Returns
    -------
    w : (N) double ndarray
        The approximation at ti+h
    """
    
    F = lambda x : x - wi - h*f(ti+h/2, (wi + x)/2)
    guess  = wi + h*f(ti,wi)
    w = op.fsolve(F, guess) 
    return w
    
def RK4Step(f, wi, ti, h):
    """
    Runge-Kutta 4 step for approximating ordinary differential equations.
    The step is given by::
        w = wi + h/6*(S0 + 2*S1 + 2*S2 + S3)
    Where::
        S0 = f(ti, wi)
        S1 = f(ti + h/2, wi + h/2*S0)
        S2 = f(ti + h/2, wi + h/2*S1)
        S3 = f(ti + h, wi + h*S2)
    This method has a local error of O(h^5), and a global error of O(h^4).
    
    Parameters
    ----------
    f : function
        Right hand side of the ordinary differential equation.
    y0 : (N) double ndarray
        The initial values for y.
    h : double
        The step taken.
    
    Returns
    -------
    w : (N) double ndarray
        The approximation at ti+h
    """
    
    S0 = f(ti , wi)
    S1 = f(ti + h/2, wi + h/2*S0)
    S2 = f(ti + h/2, wi + h/2*S1)
    S3 = f(ti + h, wi + h*S2)
    w = wi + h/6*(S0+2*S1+2*S2+S3)
    
    return w

def RK23Step(f, wj, tj, hj, tol, maxStep=np.inf):
    """
    Embedded RK2/3 step. UNFINISHED
    TODO
        Smart step resizing
        Higher dimmension for wj
        Substitute recursion with iteration
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
    """
    Solves an initial problem value::
        y'(t)=f(t,y); y(t0)=y0
    For all t in the interval I = (t0, tf). Note that the solution:
        Exists if f is continuous with respect to y in (t0,y0)
        Is unique in some interval around y0 if it is Lipschitz continuous in y0
    The solver can use different methods in order to approximate the ordinary differential equations.
    The currently supported methods are::
        ==================  ==========================
        Method              Aliases
        ==================  ==========================
        Explicit Euler      'Explicit Euler', 'EE'
        Implicit Euler      'Implicit Euler', 'IE'
        Explicit Trapezoid  'Explicit Trapezoid', 'ET'
        Implicit Trapezoid  'Implicit Trapezoid', 'IT'
        Explicit Midpoint   'Explicit Midpoint', 'EM'
        Implicit Midpoint   'Implicit Midpoint', 'IM'
        Runge-Kutta 4       'Runge-Kutta 4', 'RK4'
        Runge-Kutta 2/3     'Runge-Kutta 2/3', 'RK23'
        ==================  ==========================
    
    Parameters
    ----------
    f : function
        Right hand side of the ordinary differential equation.
    y0 : (N) double ndarray
        The initial values for y.
    I : tuple (t0 tf)
        The interval (t0, tf) over which to approximate the solution.
    m : int, optional
        Number of steps in the approximation. Default is 1e3.
    method : string, optional
        Method to use. Case insensitive. Default is 'RK4'.
    tol : double, optional
        Relative tolerance criteria for dynamic step methods. Default is 1e-4.
    maxStep : double, optional
        Maximum step size for dynamic step methods. Default is np.inf.
    
    
    Returns
    -------
    T : (m+1) double ndarray
        The values of t at which the approximations were calculated. Regularly spaced for static step methods.
    W : (N, m+1) double ndarray
        The approximations of y(T).
    """
    
    h, T, W = initODE(y0, I, m)
    method = method.casefold()
    if method in ['explicit euler', 'ee']:
        
        for i in range(m):
            W[:,i+1]=EEStep(f, W[:,i], T[i], h)
        
    elif method in ['implicit euler', 'ie']:
        
        for i in range(m):
            W[:,i+1]=IEStep(f, W[:,i], T[i], h)
        
    elif method in ['explicit trapezoid', 'et']:
        
        for i in range(m):
            W[:,i+1]=ETStep(f, W[:,i], T[i], h)
        
    elif method in ['implicit trapezoid', 'it']:
        
        for i in range(m):
            W[:,i+1]=ITStep(f, W[:,i], T[i], h)
        
    elif method in ['explicit midpoint', 'em']:
        
        for i in range(m):
            W[:,i+1]=EMPStep(f, W[:,i], T[i], h)
        
    elif method in ['implicit midpoint', 'im']:
        
        for i in range(m):
            W[:,i+1]=IMPStep(f, W[:,i], T[i], h)
        
    elif method in ['runge-kutta 4', 'rk4']:
        
        for i in range(m):
            W[:,i+1]=RK4Step(f, W[:,i], T[i], h)
        
    elif method in ['runge-kutta 2/3', 'rk23']:
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