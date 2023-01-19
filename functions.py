from configs import *

#w = lambda z: E*z*(1-z)
#O = lambda z: (1-1j)/2*np.sqrt(q/w(z))
@numba.njit
def w(z):
    return E*z*(1-z)

@numba.njit
def O(z):
    return (1-1j)/2*np.sqrt(q/w(z))

# Calculate the potential matrix
@numba.njit()
def compute_V(a11,a12,a21,a22,z):
    pre=-q/(4*CF)
    for i1, u1t in enumerate(u1):
        for i2, u2t in enumerate(u2):
            for j1, v1t in enumerate(v1):
                for j2, v2t in enumerate(v2):
                    a11[i1][i2][j1][j2]=pre*(CF*(u1t**2+v1t**2+u2t**2+v2t**2)+1/Nc*(u1t*v1t+u2t*v2t))
                    a12[i1][i2][j1][j2]=pre*(-(u1t*v1t+u2t*v2t)/Nc)
                    a21[i1][i2][j1][j2]=pre*(Nc*z*(1-z)*((u1t-v1t)**2+(u2t-v2t)**2))
                    a22[i1][i2][j1][j2]=pre*(CF-z*(1-z)*Nc)*((u1t-v1t)**2+(u2t-v2t)**2)
    return a11,a12,a21,a22



# Calculate the potential matrix in the large-Nc
@numba.njit()
def compute_VNc(a11,a12,a21,a22,z):
    pre=-q/4
    for i1, u1t in enumerate(u1):
        for i2, u2t in enumerate(u2):
            for j1, v1t in enumerate(v1):
                for j2, v2t in enumerate(v2):
                    a11[i1][i2][j1][j2]=pre*(u1t**2+v1t**2+u2t**2+v2t**2)
                    a12[i1][i2][j1][j2]=0
                    a21[i1][i2][j1][j2]=pre*(2*z*(1-z)*((u1t-v1t)**2+(u2t-v2t)**2))
                    a22[i1][i2][j1][j2]=pre*(1-2*z*(1-z))*((u1t-v1t)**2+(u2t-v2t)**2)
    return a11,a12,a21,a22


# Make nonhomogenous term in the Sch eq
@numba.njit(fastmath=True)
def compute_nonhom(psi0,t,p1,p2,z):
    pre = -w(z)/np.pi
    for i1, u1t in enumerate(u1):
        for i2, u2t in enumerate(u2):
            for j1, v1t in enumerate(v1):
                for j2, v2t in enumerate(v2):
                    frac = 1/(u1t**2+u2t**2)
                    delt = u1t*ddelta[j1]*delta[j2]+u2t*delta[j1]*ddelta[j2]
                    f_phase = -1j*(p1*(u1t-v1t)+p2*(u2t-v2t))
                    k3_exp = 1j*w(z)*O(z)*(u1t**2+u2t**2)/(2*np.tan(O(z)*t))
 
                    psi0[i1,i2,j1,j2] = pre*frac*delt*np.exp(f_phase)*np.exp(k3_exp)
    return psi0


# Simple integrater, that can be changed to a more fancy one
def integrate(arr):
    return arr.sum()*du1*du2*dv1*dv2

# The expected value of the large-Nc
def fasit1Nc(t,L,p1,p2,z):
    pre= -2*1j*w(z)/np.cosh(O(z)*t)**2
    
    expo1 = 1j*np.tanh(O(z)*t)/(2*w(z)*O(z))*(p1**2+p2**2)
    expo2 = -1j*np.tan(O(z)*t)/(2*w(z)*O(z))*(p1**2+p2**2)
    expo3 = -1j*np.tan(O(z)*L)/(2*w(z)*O(z))*(p1**2+p2**2)
    
    return pre*np.exp(expo1)*(np.exp(expo2)-np.exp(expo3))

# The expected value of the large-Nc, integrated over time
def fasit1Ncint(L,p1,p2,z):
    def real_fas(t,L,p1,p2,z):
        return np.real(fasit1Nc(t,L,p1,p2,z))
    def imag_fas(t,L,p1,p2,z):
        return np.imag(fasit1Nc(t,L,p1,p2,z))
    re = quad(real_fas,0,L,args=(L,p1,p2,z))[0]
    im = quad(imag_fas,0,L,args=(L,p1,p2,z))[0]
    
    return re + 1j*im


def fasit2Ncdiag(t,L,p1,p2,z):
    pre= -2*1j*w(z)
    
    num = 2*w(z)*O(z)/np.tan(O(z)*t)
    den = 2*w(z)*O(z)/np.tan(O(z)*t)+1j*q*(z**2+(1-z)**2)*(L-t)
    
    return pre*(1-num/den*np.exp(-1j*(p1**2+p2**2)/den))

def fasit2Ncdiagint(L,p1,p2,z):
    def real_fas(t,L,p1,p2,z):
        return np.real(fasit2Ncdiag(t,L,p1,p2,z))
    def imag_fas(t,L,p1,p2,z):
        return np.imag(fasit2Ncdiag(t,L,p1,p2,z))
    re = quad(real_fas,0,L,args=(L,p1,p2,z))[0]
    im = quad(imag_fas,0,L,args=(L,p1,p2,z))[0]
    
    return re + 1j*im


# Resets the values of every array involved in the RK
def refresh():    
    k1 = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    k2 = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    k3 = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    
    l1 = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    l2 = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    l3 = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')

    psi1next = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    psi2next = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    
    psi_nonhom = np.zeros([Nu1,Nu2,Nv1,Nv2],dtype = 'complex_')
    
    return k1,k2,k3,l1,l2,l3,psi1next,psi2next,psi_nonhom


# The Schrodinger equation itself
@numba.njit(fastmath = True)
def diff(F,Fother,V,Vother,k,l,i1,i2,j1,j2,h,nonhom,z):
    u1der = F[i1+1,i2,j1,j2]+h*k[i1+1,i2,j1,j2] - 2*(F[i1,i2,j1,j2]+h*k[i1,i2,j1,j2]) + F[i1-1,i2,j1,j2]+h*k[i1-1,i2,j1,j2]
    u2der = F[i1,i2+1,j1,j2]+h*k[i1,i2+1,j1,j2] - 2*(F[i1,i2,j1,j2]+h*k[i1,i2,j1,j2]) + F[i1,i2-1,j1,j2]+h*k[i1,i2-1,j1,j2]
    v1der = F[i1,i2,j1+1,j2]+h*k[i1,i2,j1+1,j2] - 2*(F[i1,i2,j1,j2]+h*k[i1,i2,j1,j2]) + F[i1,i2,j1-1,j2]+h*k[i1,i2,j1-1,j2]
    v2der = F[i1,i2,j1,j2+1]+h*k[i1,i2,j1,j2+1] - 2*(F[i1,i2,j1,j2]+h*k[i1,i2,j1,j2]) + F[i1,i2,j1,j2-1]+h*k[i1,i2,j1,j2-1]
    pot  = V[i1,i2,j1,j2]*(F[i1,i2,j1,j2]+h*k[i1,i2,j1,j2])
    potother = Vother[i1,i2,j1,j2]*(Fother[i1,i2,j1,j2]+h*l[i1,i2,j1,j2])
    noho = nonhom[i1,i2,j1,j2]
    
    return 1j/(2*w(z))*(u1der/du1**2+u2der/du2**2-v1der/dv1**2-v2der/dv2**2)+pot+potother-1j*noho

# This solves one time step of the Sch eq through Runge-Kutta4
@numba.njit(fastmath = True, parallel = True)
def compute_psi_runge(psi1,psi2,psi1next,psi2next,k1,k2,k3,l1,l2,l3,nonhom,t,p1,p2,z,a11,a12,a21,a22):
    h=0
    nonhom = compute_nonhom(nonhom,t+h,p1,p2,z)
    for i1 in prange(1, Nu1-1):
        for i2 in prange(1, Nu1-1):
            for j1 in prange(1, Nu1-1):
                for j2 in prange(1, Nu1-1):
                    k1[i1,i2,j1,j2]=diff(psi1,psi2,a11,a12,k0,l0,i1,i2,j1,j2,h,nonhom,z)
                    l1[i1,i2,j1,j2]=diff(psi2,psi1,a22,a21,l0,k0,i1,i2,j1,j2,h,nonhom,z)
                    
    h=dt/2
    nonhom = compute_nonhom(nonhom,t+h,p1,p2,z)
    for i1 in prange(1, Nu1-1):
        for i2 in prange(1, Nu1-1):
            for j1 in prange(1, Nu1-1):
                for j2 in prange(1, Nu1-1):
                    k2[i1,i2,j1,j2]=diff(psi1,psi2,a11,a12,k1,l1,i1,i2,j1,j2,h,nonhom,z)
                    l2[i1,i2,j1,j2]=diff(psi2,psi1,a22,a21,l1,k1,i1,i2,j1,j2,h,nonhom,z)

    h=dt/2
    nonhom = compute_nonhom(nonhom,t+h,p1,p2,z)
    for i1 in prange(1, Nu1-1):
        for i2 in prange(1, Nu1-1):
            for j1 in prange(1, Nu1-1):
                for j2 in prange(1, Nu1-1):
                    k3[i1,i2,j1,j2]=diff(psi1,psi2,a11,a12,k2,l2,i1,i2,j1,j2,h,nonhom,z)
                    l3[i1,i2,j1,j2]=diff(psi2,psi1,a22,a21,l2,k2,i1,i2,j1,j2,h,nonhom,z)
                    
    h=dt
    nonhom = compute_nonhom(nonhom,t+h,p1,p2,z)                
    for i1 in prange(1, Nu1-1):
        for i2 in prange(1, Nu1-1):
            for j1 in prange(1, Nu1-1):
                for j2 in prange(1, Nu1-1):
                    k4=diff(psi1,psi2,a11,a12,k3,l3,i1,i2,j1,j2,h,nonhom,z)
                    l4=diff(psi2,psi1,a22,a21,l3,k3,i1,i2,j1,j2,h,nonhom,z)
                    
                    psi1next[i1,i2,j1,j2] = psi1[i1,i2,j1,j2]+dt/6*(k1[i1,i2,j1,j2]+2*k2[i1,i2,j1,j2]+2*k3[i1,i2,j1,j2]+k4)
                    psi2next[i1,i2,j1,j2] = psi2[i1,i2,j1,j2]+dt/6*(l1[i1,i2,j1,j2]+2*l2[i1,i2,j1,j2]+2*l3[i1,i2,j1,j2]+l4)
    
    return psi1next,psi2next

# This solves one time step of the Sch eq through Runge-Kutta4, for the large-Nc
# This function is probably redundant
'''@numba.njit(fastmath = True, parallel = True)
def compute_psi_runge_Nc(psi1,psi2,psi1next,psi2next,k1,k2,k3,l1,l2,l3,nonhom,t,p1,p2,z):
    h=0
    nonhom = compute_nonhom(nonhom,t+h,p1,p2,z)
    for i1 in prange(1, Nu1-1):
        for i2 in prange(1, Nu1-1):
            for j1 in prange(1, Nu1-1):
                for j2 in prange(1, Nu1-1):
                    k1[i1,i2,j1,j2]=diff(psi1,psi2,a11Nc,a12Nc,k0,l0,i1,i2,j1,j2,h,nonhom,z)
                    l1[i1,i2,j1,j2]=diff(psi2,psi1,a22Nc,a21Nc,l0,k0,i1,i2,j1,j2,h,nonhom,z)
                    
    h=dt/2
    nonhom = compute_nonhom(nonhom,t+h,p1,p2,z)
    for i1 in prange(1, Nu1-1):
        for i2 in prange(1, Nu1-1):
            for j1 in prange(1, Nu1-1):
                for j2 in prange(1, Nu1-1):
                    k2[i1,i2,j1,j2]=diff(psi1,psi2,a11Nc,a12Nc,k1,l1,i1,i2,j1,j2,h,nonhom,z)
                    l2[i1,i2,j1,j2]=diff(psi2,psi1,a22Nc,a21Nc,l1,k1,i1,i2,j1,j2,h,nonhom,z)

    h=dt/2
    nonhom = compute_nonhom(nonhom,t+h,p1,p2,z)
    for i1 in prange(1, Nu1-1):
        for i2 in prange(1, Nu1-1):
            for j1 in prange(1, Nu1-1):
                for j2 in prange(1, Nu1-1):
                    k3[i1,i2,j1,j2]=diff(psi1,psi2,a11Nc,a12Nc,k2,l2,i1,i2,j1,j2,h,nonhom,z)
                    l3[i1,i2,j1,j2]=diff(psi2,psi1,a22Nc,a21Nc,l2,k2,i1,i2,j1,j2,h,nonhom,z)
                    
    h=dt
    nonhom = compute_nonhom(nonhom,t+h,p1,p2,z)                
    for i1 in prange(1, Nu1-1):
        for i2 in prange(1, Nu1-1):
            for j1 in prange(1, Nu1-1):
                for j2 in prange(1, Nu1-1):
                    k4=diff(psi1,psi2,a11Nc,a12Nc,k3,l3,i1,i2,j1,j2,h,nonhom,z)
                    l4=diff(psi2,psi1,a22Nc,a21Nc,l3,k3,i1,i2,j1,j2,h,nonhom,z)
                    
                    psi1next[i1,i2,j1,j2] = psi1[i1,i2,j1,j2]+dt/6*(k1[i1,i2,j1,j2]+2*k2[i1,i2,j1,j2]+2*k3[i1,i2,j1,j2]+k4)
                    psi2next[i1,i2,j1,j2] = psi2[i1,i2,j1,j2]+dt/6*(l1[i1,i2,j1,j2]+2*l2[i1,i2,j1,j2]+2*l3[i1,i2,j1,j2]+l4)
    
    return psi1next,psi2next
'''
