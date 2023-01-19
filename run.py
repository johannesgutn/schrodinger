from main import *

# Which values should we use for p [GeV] and z
points = 10

lower_z = 0.01
upper_z = 0.5
log_oneover_z = np.linspace(np.log(1/lower_z),np.log(1/upper_z),points)
z_values = np.exp(-log_oneover_z)

# Use theta as proxy for p, as p ~ z*(1-z)*th*E
lower_th = 0.01
upper_th = 0.5 
log_oneover_th = np.linspace(np.log(1/lower_th),np.log(1/upper_th),points)
theta_values = np.exp(-log_oneover_th)

for z in z_values:
    p_values = theta_values*z*(1-z)*EGev
    
    # Computing the potential matrices
    a11,a12,a21,a22 = compute_V(a11,a12,a21,a22,z)
    a11Nc,a12Nc,a21Nc,a22Nc = compute_VNc(a11Nc,a12Nc,a21Nc,a22Nc,z)
    a21Ncdiag = np.zeros([Nu1,Nu2,Nv1,Nv2])

    # Solves the Sch eq for the p and z values above
    for p in p_values:
        # Check whether we have already done this calculation
        if not os.path.exists(f'{filename}_p={p:.3f}_z={z:.3f}_L={tmax}_E={EGev}_gridpoints={N}_gridsize={ma}.npy'):
            main(p,z,a11,a12,a21,a22,a11Nc,a12Nc,a21Nc,a22Nc,a21Ncdiag)