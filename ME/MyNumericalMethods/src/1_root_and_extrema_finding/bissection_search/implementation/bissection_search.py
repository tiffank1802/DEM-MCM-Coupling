import numpy as np
from typing import Callable

# Definition de la fonction

def bissection_search(
    f: Callable[[float],float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iterations: int = 1_000,
)-> float:
    fa: float = f(a)
    fb: float =f(b)
    
    # Check if either endpoint is a root
    if np.abs(fa)<tol:
        return a
    if np.abs(fb)<tol:
        return b
    if fa*fb>0:
        raise ValueError('Funtion must have opposite signs at endpoints a and b.')
    for _ in range(max_iterations):
        c: float=(a+b)/2
        fc: float=f(c)
        
        if np.abs(fc)<tol or (b-a)/2<tol:
            return c
        if fa*fc<0:
            b,fb=c,fc
        else:
            a,fa=c,fc
            
    raise ValueError(
        "Bissection method did not converge within the number of iterations."
    )