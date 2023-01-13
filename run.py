from main import *

# Which values should we use for p [GeV] and z
p_values =[0.4]
z_values = [0.3]

for z in z_values:
    # Computing the potential matrices
    compute_V(a11,a12,a21,a22,z)
    compute_VNc(a11Nc,a12Nc,a21Nc,a22Nc,z)

    if diag:
        a21Nc= np.zeros([Nu1,Nu2,Nv1,Nv2])

    # Solves the Sch eq for the p and z values above
    for p in p_values:
        # Check whether we have already done this calculation
        if not os.path.exists(f'{filename}_p={p}_z={z}_L={tmax}_E={EGev}_gridpoints={N}_gridsize={ma}.npy'):
            main(p,z)