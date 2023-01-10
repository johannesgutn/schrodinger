from functions import *

# Computing the potential matrices
compute_V(a11,a12,a21,a22)
compute_VNc(a11Nc,a12Nc,a21Nc,a22Nc)

if diag:
    a21Nc= np.zeros([Nu1,Nu2,Nv1,Nv2])

# Which values should we use for p [GeV] and theta 
p_values =[0.1,0.2]
theta_values = [0.1,0.2]
# Solves the Sch eq for these values and saves the result
for p in p_values:
    for theta in theta_values:
        # Check whether we have already done this calculation
        if not os.path.exists(f'{filename}_p={p}_theta={theta}_gridpoints={N}_gridsize={ma}_L={tmax}_E={EGev}_z={z}.npy'):
            main(p,theta)