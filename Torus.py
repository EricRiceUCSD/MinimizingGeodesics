import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ApproximationFuncs import *

r = 1           # Inner radius
R = 2           # Outer radius

p = np.array([0, 0])
q = np.array([3*np.pi/4, 2*np.pi/3])
#guessVector = np.array([0, 1])

def X(x):
    """The coordinate chart on T^2,
    where x = (x[0], x[1]) \in [0,2\pi)^2."""
    return np.array([(R + r*np.cos(x[0]))*np.cos(x[1]),
                     (R + r*np.cos(x[0]))*np.sin(x[1]),
                     r*np.sin(x[0])])

def dX(x):
    return np.array([[-r*np.sin(x[0])*np.cos(x[1]),\
                          -(R + r*np.cos(x[0]))*np.sin(x[1])],
                     [-r*np.sin(x[0])*np.sin(x[1]),\
                          (R + r*np.cos(x[0]))*np.cos(x[1])],
                     [r*np.cos(x[0]), 0]])

def g(*args):
    """The matrix of the metric tensor on T^2.  Takes an optional
    argument in the form of an array x = (x[0], x[1]), in which
    case g(x) is the matrix based at the point x."""
    if len(args) > 0:
        return np.matrix([[r**2, 0],
                          [0, (R + r*np.cos(args[0][0]))**2]])
    else:
        def g_11(x):
            return (R + r*np.cos(x[0]))**2
        return np.matrix([[lambda _: r**2, lambda _: 0],
                          [lambda _: 0, g_11]])

def geoField(z):
    if len(z) != 4:
        return None

    p = np.array([z[0], z[1]])
    Gamma = Christoffel(g, p)
    flow = np.zeros(4)
    for k in range(0,2):
        flow[k] = z[k + 2]
        summ = 0
        for i in range(0,2):
            for j in range(0,2):
                summ += -Gamma[i,j,k]*z[i + 2]*z[j + 2]
        flow[k + 2] = summ

    return flow

def exp(p, v):
    return RK4(geoField, 0.1, np.array([p[0], p[1], v[0], v[1]]), 10)[-1, 0:2]

def F(*args):
    """The difference between exp(p, v) and q.  Optional argument allows
    one to specify v."""
    if len(args) > 0:
        return exp(p, args[0]) - q
    else:
        def F_0(v):
            return (exp(p, v) - q)[0]
        def F_1(v):
            return (exp(p, v) - q)[1]
        return np.array([F_0, F_1])

# Determining a reasonable guess vector
v1, v2, v3, v4 = np.array([1, 0]), np.array([0, 1]),\
                 np.array([-1, 0]), np.array([0, -1])
ev1, ev2, ev3, ev4 = exp(p, np.matmul(dX(p), v1)),\
                     exp(p, np.matmul(dX(p), v2)),\
                     exp(p, np.matmul(dX(p), v3)),\
                     exp(p, np.matmul(dX(p), v4))
guesses = [v1, v2, v3, v4]
eguesses = [ev1, ev2, ev3, ev4]

ind = 0
for i in range(1,4):
    if EuclidNorm(eguesses[i] - q) < EuclidNorm(eguesses[ind] - q):
        ind = i

guessVector = guesses[i]

# Computing approximate v
v = NewtonsMethod(F, guessVector, 0.01)

# Preparing 3D Graph
fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig)
ax.text2D(0.05, 0.95, "The approximate minimizing geodesic of $T^2$ between $p$ and $q$",\
          transform=ax.transAxes)
ax.set_xlim3d(-3, 3)
ax.set_ylim3d(-3, 3)
ax.set_zlim3d(-3, 3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Plots the Torus, for reference.
s, t = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]
Z = X(np.array([s, t]))
x, y, z = Z[0], Z[1], Z[2]
ax.plot_surface(x, y, z, color=(0.2, 0.2, 0.3, 0.2))

# Plots the flow of the approximate minimizing geodesic between p and q.
N = 100
coordFlow = RK4(geoField, 0.01, np.array([p[0], p[1], v[0], v[1]]), N)
maniFlow = [X(coordFlow[i][0:2]) for i in range(0,N)]
xdata, ydata, zdata = [], [], []
for i in range(0,N):
    xdata += [maniFlow[i][0]]
    ydata += [maniFlow[i][1]]
    zdata += [maniFlow[i][2]]
ax.plot(xdata, ydata, zdata)

# Plotting p and q
P, Q = X(p), X(q)
ax.scatter([P[0], Q[0]], [P[1], Q[1]], [P[2], Q[2]], color=(1, 0, 0))
ax.text(P[0], P[1], P[2], '%s' % ('$p$'), size=20, color='k')
ax.text(Q[0], Q[1], Q[2], '%s' % ('$q$'), size=20, color='k')



plt.show()
