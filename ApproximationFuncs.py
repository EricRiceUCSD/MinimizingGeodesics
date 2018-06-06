import numpy as np
import math
import matplotlib.pyplot as plt

def RK4(sys, h, init, n):
    """Given a system of ODEs sys, time step h, and initial array init,
    computes the first n approximations of the flow of init along its
    trajectory of sys, separated by time h, via the RK4 method."""

    z = [init]

    for j in range(0, n):
        Z1 = z[j]
        Z2 = z[j] + (h/2)*sys(Z1)
        Z3 = z[j] + (h/2)*sys(Z2)
        Z4 = z[j] + h*sys(Z3)
        z += [z[j] + (h/6)*(sys(Z1) + 2*sys(Z2) + 2*sys(Z3) + sys(Z4))]

    return np.asarray(z)

def partial(f, k, x, h=0.0001):
    """Computes the partial derivative of a function f : R^n \to R
    in the kth coordinate variable evaluated at a point x in R^n,
    using a finite difference with (optional) step size h."""

    n = len(x)
    dx_k = np.zeros(n)
    dx_k[k] = h/2

    return (f(x+dx_k)-f(x-dx_k))/h

def jacobian(f, x):
    """Computes the Jacobian of a map f : R^n \to R^m evaluated at a
    point x in R^n.  f must be given as an array of functions R^n \to R."""

    n = len(x)
    m = len(f())
    J = np.zeros((m,n))
    for i in range(0,m):
        for j in range(0,n):
            J[i,j] = partial(f()[i], j, x)

    return J
            
    
def Christoffel(g, x):
    """Returns an array whose [i,j,k]-entry is the Christoffel symbol
    Gamma_{ij}^k based at the point with coordinate representation x,
    under the Levi-Civita connection.  The input g should be a matrix
    of functions representing the metric tensor, with g(x) being the
    matrix on T_xM."""
    
    n = len(x)                          # dimension of manifold
    ginv = np.linalg.inv(g(x))          # inverse matrix of g(x)
    Gamma = np.zeros((n,n,n))
    
    # We make use of the symmetry of the Levi-Civita connection
    # in this computation.
    for k in range(0,n):
        for i in range(0,n):
            for j in range(i,n):
                summ = 0
                for m in range(0,n):
                     summ += 0.5*ginv[k,m]*(partial(g()[i,m], j, x)\
                            + partial(g()[j,m], i, x)\
                            - partial(g()[i,j], m, x))
                Gamma[i,j,k] = summ

    return Gamma

def EuclidNorm(x):
    n = len(x)
    summ = 0
    for i in range(0,n):
        summ += x[i]**2
    return np.sqrt(summ)

def NewtonsMethod(F, guess, error):
    """Looks to find a root of a function F : R^n \to R^m,
    given an initial guess and error tolerance.  F must be
    given as an array of functions R^n \to R."""
    x = [guess]
    k = 0
    while EuclidNorm(F(x[k])) >= error:
        x += [x[k] - np.matmul(np.linalg.inv(jacobian(F, x[k])),\
                                 F(x[k]))]
        k += 1
        if k > 100:
            return None

    return x[-1]
    
