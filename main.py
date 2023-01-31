from functions import *

'''
We want: 1finite, 1Nc, 2finite, 2Nc, 2Nc-diag
The most efficient way to do it is to first compute 1Nc and 2Nc-diag trough separate diff eqs, then use 1Nc to compute 2Nc. Then at last compute 1finite and 2finite.
We know the correct answer for 1Nc and 2Nc-diag
We can to keep the real part
'''


# Solves the Sch eq for finite Nc and large-Nc
# Saves the results to be analyzed
# Could be more efficient, as it solves the Sch eq 3 times, and we repeat the same notation 3 times
def main(p_GeV,z,a11,a12,a21,a22,a11Nc,a12Nc,a21Nc,a22Nc,a21Ncdiag):
    start = timer()
    

    # Convert p to fm^-1 and transform to cartesian coordinates
    p = p_GeV * 5.076
    phi = 0  # This constitutes a rotation in the transverse plane, which is symmetric. The value should be arbitrary, so we set it to zero.
    p1 = p*np.cos(phi)
    p2 = p*np.sin(phi)

    theta = p/w(z)
    
    print(f'theta={theta:.3}, z={z:.3}')

    ###################################################################
    # Calculate diagonal Nc first
    normals_runge1Ncdiag = np.array([0]*Nt).astype(complex)
    normals_runge2Ncdiag = np.array([0]*Nt).astype(complex)

    psi1Ncdiag = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    psi2Ncdiag = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')


    for i in range(1,Nt):
        k1,k2,k3,l1,l2,l3,psi1next,psi2next,psi_nonhom = refresh()

        psi1Ncdiag,psi2Ncdiag = compute_psi_runge(psi1Ncdiag,psi2Ncdiag,psi1next,psi2next,k1,k2,k3,l1,l2,l3,psi_nonhom,i*dt,p1,p2,z,a11Nc,a12Nc,a21Ncdiag,a22Nc)

        normals_runge1Ncdiag[i] = integrate(psi1Ncdiag)
        normals_runge2Ncdiag[i] = integrate(psi2Ncdiag)

        print(f'time = {i*dt:.3f}: psi1Ncdiag is {normals_runge1Ncdiag[i]:.3f} (should be {fasit1Ncint(i*dt,p1,p2,z):.3f}), and psi2Ncdiag is {normals_runge2Ncdiag[i]:.3f} (should be {fasit2Ncdiagint(i*dt,p1,p2,z):.3f})')

        t_stop = tmax
        if np.abs(integrate(psi1Ncdiag))>1000 or np.abs(integrate(psi2Ncdiag))>1000:
            t_stop = i*dt
            break

    ###################################################################
    # Then calculate large-Nc. The only difference is setting the non-diagonal part of the potential matrix to zero
    normals_runge1Nc = np.array([0]*Nt).astype(complex)
    normals_runge2Nc = np.array([0]*Nt).astype(complex)

    psi1Nc = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    psi2Nc = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')


    for i in range(1,Nt):
        k1,k2,k3,l1,l2,l3,psi1next,psi2next,psi_nonhom = refresh()

        psi1Nc,psi2Nc = compute_psi_runge(psi1Nc,psi2Nc,psi1next,psi2next,k1,k2,k3,l1,l2,l3,psi_nonhom,i*dt,p1,p2,z,a11Nc,a12Nc,a21Nc,a22Nc)

        normals_runge1Nc[i] = integrate(psi1Nc)
        normals_runge2Nc[i] = integrate(psi2Nc)

        print(f'time = {i*dt:.3f}: psi1Nc is {normals_runge1Nc[i]:.3f} (should be {fasit1Ncint(i*dt,p1,p2,z):.3f}), and psi2Nc is {normals_runge2Nc[i]:.3f} (Nc-diag is {fasit2Ncdiagint(i*dt,p1,p2,z):.3f})')

        if np.abs(integrate(psi1Nc))>1000 or np.abs(integrate(psi2Nc))>1000:
            break


    ###################################################################
    # Then calculate finite Nc
    normals_runge1 = np.array([0]*Nt).astype(complex)
    normals_runge2 = np.array([0]*Nt).astype(complex)

    psi1 = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    psi2 = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')

    for i in range(1,Nt):
        k1,k2,k3,l1,l2,l3,psi1next,psi2next,psi_nonhom = refresh()
                
        psi1,psi2 = compute_psi_runge(psi1,psi2,psi1next,psi2next,k1,k2,k3,l1,l2,l3,psi_nonhom,i*dt,p1,p2,z,a11,a12,a21,a22)
        
        normals_runge1[i] = integrate(psi1)
        normals_runge2[i] = integrate(psi2)
        
        print(f'time = {i*dt:.3f}: psi1 is {normals_runge1[i]:.3f} (Nc is {fasit1Ncint(i*dt,p1,p2,z):.3f}), and psi2 is {normals_runge2[i]:.3f} (Nc-diag is {fasit2Ncdiagint(i*dt,p1,p2,z):.3f})')

        if np.abs(integrate(psi1))>1000 or np.abs(integrate(psi2))>1000:
            break

    ###################################################################
    fasit1Nc_array = np.zeros([Nt]).astype(complex)
    for i, s in enumerate(t):
        if i > 0:
            fasit1Nc_array[i]=fasit1Ncint(s,p1,p2,z)

    fasit2Ncdiag_array = np.zeros([Nt]).astype(complex)
    for i, s in enumerate(t):
        if i > 0:
            fasit2Ncdiag_array[i]=fasit2Ncdiagint(s,p1,p2,z)

    # Take the time of execution
    end = timer()
    total_time = timedelta(seconds=end-start)
    
    # Save as a numpy file
    combined = np.vstack((t,normals_runge1,normals_runge2,normals_runge1Nc,normals_runge2Nc,normals_runge1Ncdiag,normals_runge2Ncdiag,fasit1Nc_array,fasit2Ncdiag_array))
    np.save(f'{filename}_theta={theta:.3f}_z={z:.3f}_L={tmax}_E={EGev}_gridpoints={N}_gridsize={ma}.npy', combined)

    # Update the summary file
    df = pd.read_csv('data_files/sch_summary.csv')
    if t_stop == tmax:
        
        error1 = np.round(abs((fasit1Nc_array[Nt-1]-normals_runge1Nc[Nt-1])/normals_runge1Nc[Nt-1]),3)
        error2 = np.round(abs((fasit2Ncdiag_array[Nt-1]-normals_runge2Ncdiag[Nt-1])/normals_runge2Ncdiag[Nt-1]),3)

        frac_Nc_finite1 = round(np.real(normals_runge1Nc[Nt-1])/np.real(normals_runge1[Nt-1]),2)
        frac_Nc_finite2 = round(np.real(normals_runge2Nc[Nt-1])/np.real(normals_runge2[Nt-1]),2)
        frac_diag_finite2 = round(np.real(normals_runge2Ncdiag[Nt-1])/np.real(normals_runge2[Nt-1]),2)       

        finite_Nc1,large_Nc1,true_Nc1 = np.round(normals_runge1[Nt-1],3),np.round(normals_runge1Nc[Nt-1],3),np.round(fasit1Nc_array[Nt-1],3)
        finite_Nc2,large_Nc2,diag_Nc2, true_diag2 = np.round(normals_runge2[Nt-1],3),np.round(normals_runge2Nc[Nt-1],3),np.round(normals_runge2Nc[Nt-1],3),np.round(fasit2Ncdiag_array[Nt-1],3)

        df.loc[len(df)] = [round(np.real(p_GeV),3),round(p_GeV/(w(z)/5.076),3),round(np.real(z),3),np.real(tmax),np.real(EGev),np.round(w(z)/5.076,2),np.real(N),np.real(ma),finite_Nc1,large_Nc1,true_Nc1,finite_Nc2,large_Nc2,diag_Nc2,true_diag2,frac_Nc_finite1,frac_Nc_finite2,frac_diag_finite2,error1,error2,total_time]
    else:
        df.loc[len(df)] = [round(np.real(p_GeV),3),round(p_GeV/(w(z)/5.076),3),round(np.real(z),3),tmax,EGev,np.round(w(z)/5.076,2),N,ma,'failed','failed','failed','failed','failed','failed','failed','failed','failed','failed','failed','failed',total_time]

    
    df = df.sort_values(['ω','z','p'], ascending=[True,True,True])
    df.to_csv('data_files/sch_summary.csv', index=False)


    df = pd.read_csv('data_files/fails.csv')
    df.loc[len(df)] = [np.round(w(z)/5.076,2),round(np.real(z),3),np.real(EGev),round(np.real(p_GeV),3),np.real(N),np.real(ma),np.real(tmax),round(t_stop,3),t_stop==tmax]
    df = df.sort_values('t_failed', ascending=False)
    df.to_csv('data_files/fails.csv', index=False)
