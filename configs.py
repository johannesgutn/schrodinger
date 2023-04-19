import numpy as np
import os
import numba
from numba import jit,njit, prange
from scipy.integrate import quad,dblquad
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta


# Do it all in fm
# Physical parameters

qhatmix=1.5 #GeV^2/fm
q=qhatmix*25.77 #fm^(-3)
EGev = 10 # Energy in Gev
E = EGev * 5.076 # Conversion factor to fm^-1
#z=0.4

Nc=3
CF=(Nc**2-1)/(2*Nc)

#a=lambda z: np.sqrt(q/w)
#Ou=(1-1j)/2*a
#Ov=(1+1j)/2*a

# Numerical parameters
grid_size=2 # This is the grid size in fm
u1max=grid_size
u2max=grid_size
v1max=grid_size
v2max=grid_size
u1min=-u1max
u2min=-u2max
v1min=-v1max
v2min=-v2max
#tmax=1e-2
t0=0
tmax=5 # Maximum medium length in fm


N=52 # Number of grid points. Should be at least 40 for okay results, ideally more
Nu1 = N
Nu2 = N
Nv1 = N
Nv2 = N


du1 = (u1max-u1min)/(Nu1-1)
du2 = (u2max-u2min)/(Nu2-1)
dv1 = (v1max-v1min)/(Nv1-1)
dv2 = (v2max-v2min)/(Nv2-1)
hu1,hu2,hv1,hv2 = int(Nu1/2),int(Nu2/2),int(Nv1/2),int(Nv2/2)

dtdusq=0.5

deltat=dtdusq*du1**2
Nt=round((tmax-t0)/deltat) # The number of time points is decided by the griz size and number of grid points
#Nt=round(Nt/10)*10

dt = (tmax-t0)/(Nt-1)

t_stop = tmax

diag=False # If this is true you make the potential matrix fully diagonal


psi0=np.zeros([Nu1,Nu2,Nv1,Nv2]) 

# Make the grid
u1 = np.linspace(u1min, u1max, Nu1)
u2 = np.linspace(u2min, u2max, Nu2)
v1 = np.linspace(v1min, v1max, Nv1)
v2 = np.linspace(v2min, v2max, Nv2)
t = np.linspace(t0, tmax, Nt)
U1,U2,V1,V2 = np.meshgrid(u1,u2,v1,v2, indexing='ij')

# This initalizes the potential matrix
a11=np.zeros([Nu1,Nu2,Nv1,Nv2])
a12=np.zeros([Nu1,Nu2,Nv1,Nv2])
a21=np.zeros([Nu1,Nu2,Nv1,Nv2])
a22=np.zeros([Nu1,Nu2,Nv1,Nv2])

a11Nc=np.zeros([Nu1,Nu2,Nv1,Nv2])
a12Nc=np.zeros([Nu1,Nu2,Nv1,Nv2])
a21Nc=np.zeros([Nu1,Nu2,Nv1,Nv2])
a22Nc=np.zeros([Nu1,Nu2,Nv1,Nv2])


k0 = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
l0 = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')

# Make delta and derivative of delta as sharp points
delta = np.zeros(Nu1)
for i in [-1,0]:
    delta[int(Nu1/2)+i] = 1/(2*du1)

ddelta = np.zeros(Nu1)
ddelta[int(Nu1/2)-1] = 1/du1**2
ddelta[int(Nu1/2)] = -1/du1**2

filename = 'data_files/schr'