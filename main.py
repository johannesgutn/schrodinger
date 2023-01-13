from functions import *

# Solves the Sch eq for finite Nc and large-Nc
# Saves the results to be analyzed
def main(p_GeV,z):
    start = timer()
    # Convert p to fm^-1 and transform to cartesian coordinates
    p = p_GeV * 5.076
    phi = 0  # This constitutes a rotation in the transverse plane, which is symmetric. The value should be arbitrary, so we set it to zero.
    p1 = p*np.cos(phi)
    p2 = p*np.sin(phi)

 # Calculate large-Nc first
    normals_runge1Nc = np.array([0]*Nt).astype(complex)
    normals_runge2Nc = np.array([0]*Nt).astype(complex)

    psi1Nc = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    psi2Nc = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')


    for i in range(1,Nt):
        k1,k2,k3,l1,l2,l3,psi1next,psi2next,psi_nonhom = refresh()

        psi1Nc,psi2Nc = compute_psi_runge_Nc(psi1Nc,psi2Nc,psi1next,psi2next,k1,k2,k3,l1,l2,l3,psi_nonhom,i*dt,p1,p2,z)

        normals_runge1Nc[i] = integrate(psi1Nc)
        normals_runge2Nc[i] = integrate(psi2Nc)

        print(f'time = {i*dt:.3f}: psi1Nc is {normals_runge1Nc[i]:.3f} (Nc_true is {fasit1Ncint(i*dt,p1,p2,z):.3f}), and psi2Nc is {normals_runge2Nc[i]:.3f}')

        t_stop = tmax
        if np.abs(integrate(psi1Nc))>1000 or np.abs(integrate(psi2Nc))>1000:
            t_stop = i*dt
            break


    normals_runge1 = np.array([0]*Nt).astype(complex)
    normals_runge2 = np.array([0]*Nt).astype(complex)

    psi1 = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    psi2 = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')

    # Then calculate finite Nc
    for i in range(1,Nt):
        k1,k2,k3,l1,l2,l3,psi1next,psi2next,psi_nonhom = refresh()
                
        psi1,psi2 = compute_psi_runge(psi1,psi2,psi1next,psi2next,k1,k2,k3,l1,l2,l3,psi_nonhom,i*dt,p1,p2,z)
        
        normals_runge1[i] = integrate(psi1)
        normals_runge2[i] = integrate(psi2)
        
        print(f'time = {i*dt:.3f}: psi1 is {normals_runge1[i]:.3f} (Nc is {fasit1Ncint(i*dt,p1,p2,z):.3f}), and psi2 is {normals_runge2[i]:.3f}')

        if np.abs(integrate(psi1Nc))>1000 or np.abs(integrate(psi2Nc))>1000:
            t_stop = i*dt
            break

   
    fasit1Nc_array = np.zeros([Nt]).astype(complex)
    for i, s in enumerate(t):
        fasit1Nc_array[i]=fasit1Ncint(s,p1,p2,z)

    # Take the time of execution
    end = timer()
    total_time = timedelta(seconds=end-start)
    
    # Save as a numpy file
    combined = np.vstack((t,normals_runge1,normals_runge2,normals_runge1Nc,normals_runge2Nc,fasit1Nc_array))
    np.save(f'{filename}_p={p_GeV}_z={z}_L={tmax}_E={EGev}_gridpoints={N}_gridsize={ma}.npy', combined)

    # Update the summary file
    df = pd.read_csv('data_files/sch_summary.csv')
    if t_stop == tmax:
        

        error = round(abs((fasit1Nc_array[Nt-1]-normals_runge1Nc[Nt-1])/normals_runge1Nc[Nt-1]),3)

        frac1 = round(np.imag(normals_runge1Nc[Nt-1])/np.imag(normals_runge1[Nt-1]),2)
        frac2 = round(np.imag(normals_runge2Nc[Nt-1])/np.imag(normals_runge2[Nt-1]),2)

        finite_Nc1,large_Nc1,true_Nc1,finite_Nc2,large_Nc2 = np.round(normals_runge1[Nt-1],3),np.round(normals_runge1Nc[Nt-1],3),np.round(fasit1Nc_array[Nt-1],3),np.round(normals_runge2[Nt-1],3),np.round(normals_runge2Nc[Nt-1],3)

        df.loc[len(df)] = [np.real(p_GeV),np.real(z),np.real(tmax),np.real(EGev),np.round(w(z)/5.076,2),np.real(N),np.real(ma),finite_Nc1,large_Nc1,true_Nc1,finite_Nc2,large_Nc2,frac1,frac2,error,total_time]
    else:
        df.loc[len(df)] = [p_GeV,z,tmax,EGev,np.round(w(z)/5.076,2),N,ma,'failed','failed','failed','failed','failed','failed','failed','failed',total_time]

    df.to_csv('data_files/sch_summary.csv', index=False)



