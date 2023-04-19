import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import matplotlib.colors as mcolors
from matplotlib.pyplot import subplots

from functions import *

# These are the parameters for the case I want to analyze, which are not necessarily the same as those I generate

qhatmix=1.5 #GeV^2/fm
q=qhatmix*25.77 #fm^(-3)
EGev = 100 # Energy in Gev
E = EGev * 5.076 # Conversion factor to fm^-1
#z=0.4

Nc=3
CF=(Nc**2-1)/(2*Nc)

#a=lambda z: np.sqrt(q/w)
#Ou=(1-1j)/2*a
#Ov=(1+1j)/2*a

# Numerical parameters
ma=2 # This is the grid size in fm
u1max=ma
u2max=ma
v1max=ma
v2max=ma
u1min=-u1max
u2min=-u2max
v1min=-v1max
v2min=-v2max
#tmax=1e-2
t0=0
tmax=2 # Maximum medium length in fm
L=tmax


N=50 # Number of grid points. Should be at least 40 for okay results, ideally more
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
t = np.linspace(t0+0.00001, tmax, Nt)

points = 10
lower_z = 0.01
upper_z = 0.5
log_oneover_z = np.linspace(np.log(1/lower_z),np.log(1/upper_z),points)
z_values = np.exp(-log_oneover_z)

# Use theta as proxy for p, as p ~ z*(1-z)*th*E
lower_th = 0.01
upper_th = 0.5 
log_oneover_th = np.linspace(np.log(1/lower_th),np.log(1/upper_th),points)
dth = log_oneover_th[0]-log_oneover_th[1]
log_oneover_th = np.append(log_oneover_th,log_oneover_th[points-1]-dth) # Added additional th value
theta_values = np.exp(-log_oneover_th)


th_points = len(theta_values)
z_points = len(z_values)

Z, TH = np.meshgrid(log_oneover_z,log_oneover_th)

log_oneover_z_big = np.linspace(log_oneover_z[0],log_oneover_z[-1],100)
log_oneover_th_big = np.linspace(log_oneover_th[0],log_oneover_th[-1],100)

z_values_big = np.exp(-log_oneover_z_big)
theta_values_big = np.exp(-log_oneover_th_big)

def w(z):
    return E*z*(1-z)

def O(z):
    return (1-1j)/2*np.sqrt(q/w(z))

def eik2(t,L,z,th):
    pre = 4*w(z)**2/(q*(1-2*z*1*(1-z))*t**2)
    exp1 = -1j*th**2*w(z)*t/2
    exp2=-q*th**2*t**3/12
    exp3=-q*th**2*(1-2*z*1*(1-z))*(L-t)*t**2/4
    return pre*np.exp(exp1)*np.exp(exp2)*(1-np.exp(exp3))

def eik2int(L,z,th):
    def real_fas(t,L,z,th):
        return np.real(eik2(t,L,z,th))
    def imag_fas(t,L,z,th):
        return np.imag(eik2(t,L,z,th))
    re = quad(real_fas,0,L,args=(L,z,th))[0]
    im = quad(imag_fas,0,L,args=(L,z,th))[0]

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

def Fmed_inout_scal(t,z,th):
    ex = -1j*np.tan(O(z)*t)*w(z)*th**2/(2*O(z))
    return -2*np.real(1-np.exp(ex))

def Fmed_inout_eik_scal(t,z,th):
    def integrand(t1,t,z,th):
        return np.sin(w(z)*th**2*(t-t1)/2)*np.exp(-q*th**2*(t-t1)**3/12)
    return -w(z)*th**2*quad(integrand,0,t,args=(t,z,th))[0]

def Fmed_inin_scal(t,z,th):
    p1=th*w(z)
    p2=0
    return th**2/2*np.real(fasit2Ncdiagint(t,p1,p2,z))

def Fmed_inin_eik_scal(t,z,th):
    return th**2/2*np.real(eik2int(t,z,th))

def Fmed_scal(t,z,th):
    return Fmed_inin_scal(t,z,th)+Fmed_inout_scal(t,z,th)

def Fmed_eik_scal(t,z,th):
    return Fmed_inin_eik_scal(t,z,th)+Fmed_inout_eik_scal(t,z,th)

Fmed_inout = np.vectorize(Fmed_inout_scal)
Fmed_inout_eik = np.vectorize(Fmed_inout_eik_scal)

Fmed_inin = np.vectorize(Fmed_inin_scal)
Fmed_inin_eik = np.vectorize(Fmed_inin_eik_scal)

Fmed_diag = np.vectorize(Fmed_scal)
Fmed_diag_eik = np.vectorize(Fmed_eik_scal)

def round_to_significant_digits(number, significant_digits):
    if number !=0:
        # Calculate the number of decimal places to round to
        decimal_places = significant_digits - int(np.floor(np.log10(abs(number)))) - 1

        # Round the number to the specified number of decimal places
        rounded_number = round(number, decimal_places)
    else:
        rounded_number = 0

    return rounded_number

def round_sig(arr,dig):
    arr_round = arr.copy()
    for i,n in enumerate(arr_round):
        arr_round[i] = round_to_significant_digits(n,dig)
    return arr_round

def error_cal(true,approx):
    return np.abs(approx-true)/(1+np.abs(true))

def Fmed(time_frac,fmed = True,L=tmax):
    time_n = int(len(t)*time_frac)-1
    time = t[time_n]

    inin_full = np.zeros([th_points,z_points])
    inin_Nc = np.zeros([th_points,z_points])   
    inin_diag = np.zeros([th_points,z_points])
    inin_diag_true = np.zeros([th_points,z_points])
    inout_full = np.zeros([th_points,z_points])


    for i,theta in enumerate(theta_values):
        for j,z in enumerate(z_values):

            all = np.load(f'{filename}_theta={theta:.3f}_z={z:.3f}_L={L}_E={EGev}_gridpoints={N}_gridsize={grid_size}.npy')

            full2 = all[2]
            Nc2 = all[4]
            diag2 = all[6]
            diag2_fasit = all[8]

            inin_full[i,j] = theta**2/2*np.real(full2[time_n])
            inin_Nc[i,j] = theta**2/2*np.real(Nc2[time_n])
            inin_diag[i,j] = theta**2/2*np.real(diag2[time_n])
            inin_diag_true[i,j] = theta**2/2*np.real(diag2_fasit[time_n])

            inout_full[i,j] = Fmed_inout(time,z,theta)
    if not fmed:
        inout_full = 0

    Fmed_full = inin_full + inout_full
    Fmed_Nc = inin_Nc + inout_full
    Fmed_diag = inin_diag + inout_full
    Fmed_diag_true = inin_diag_true + inout_full


    return Fmed_full,Fmed_Nc,Fmed_diag,Fmed_diag_true


def pl(zvalues,cent,time,name=0,log = False,mima=0, x=TH,y=Z,show=True):

    def get_ticks(z,c,n):
        ma=np.max(z)
        mi=np.min(z)

        if type(mima)!=int:
            mi = mima[0]
            ma = mima[1]
        
        step=(ma-mi)/(n-1)
        psteps=round((ma-c)/step)
        msteps=round((c-mi)/step)
        
        ma_n = c+psteps*step
        mi_n = c-msteps*step
        ticks=np.linspace(mi_n,ma_n,n)
        lines=np.linspace(mi_n-step/2,ma_n+step/2,n+1)

        ticks = round_sig(ticks,2)
        
        return ticks, lines
    
    plt.figure(dpi=600)
    plt.rcParams['text.usetex'] = True

    fig = plt.figure(figsize=(8,6))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 

    plt.tick_params(axis='both',which='both', right=True, top=True, bottom=True, direction='in', labelsize=18)

    if log:
        from matplotlib.colors import LogNorm
        max_lvl = 2
        min_lvl = -4
        n_lvl = (max_lvl-min_lvl)+1
        levels = np.logspace(-4,2,13)
        levels = np.logspace(min_lvl,max_lvl,n_lvl)
        cp = plt.contourf(x,y,zvalues,levels=levels,cmap=plt.get_cmap('YlOrRd'),norm = LogNorm())

        def fmt(x, pos):
            a, b = '{:.0e}'.format(x).split('e')
            b = int(b)
            return r'$ 10^{{{}}}$'.format(b)

        cbar = plt.colorbar(cp, format=ticker.FuncFormatter(fmt))
        cbar.ax.tick_params(labelsize=18)

    else:
        num = 9

        ticks,lines = get_ticks(zvalues,cent,num)

        if abs(cent-lines[0]) < abs(cent-lines[num]):
            cmax = lines[num]
            cmin = 2*cent-cmax
        else:
            cmin = lines[0]
            cmax = 2*cent-cmin
  
        cp = plt.contourf(x, y,zvalues,levels=lines,vmin=cmin,vmax = cmax,cmap=plt.get_cmap('RdBu_r'))
        cbar = plt.colorbar(ticks=ticks)
        cbar.ax.tick_params(labelsize=18)

    ax.set_title(r'$\hat{q}=%.1f\,\mathrm{GeV}^2/\mathrm{fm},\, L = %.2f\,\mathrm{fm},\,E = %.0f\,\mathrm{GeV}$'% (qhatmix,time,EGev),fontsize=14)

    #ax.text(1.3,np.max(log_oneover_z)-0.25,r'$\hat{q}=%.1f\,\mathrm{GeV}^2/\mathrm{fm},\, L = %.2f\,\mathrm{fm},\,E = %.0f\,\mathrm{GeV}$'% (qhatmix,time,EGev),fontsize = 13)


    ax.set_xlabel(r'$\log 1/ \theta$', fontsize=20)
    ax.set_ylabel(r'$\log 1/z$', fontsize=20)
    
    if name !=0:
        plt.savefig(f'plots/{name}_{EGev}.png', bbox_inches='tight')
    
    if show:
        plt.show()

def time_plot(z_n,theta_n,fmed=True,L=tmax,grid_points=N,largeNc=False):
    theta = theta_values[theta_n]
    z = z_values[z_n]

    all = np.load(f'{filename}_theta={theta:.3f}_z={z:.3f}_L={L}_E={EGev}_gridpoints={grid_points}_gridsize={grid_size}.npy')
    
    t = np.real(all[0])
    full2 = all[2]
    Nc2 = all[4]
    diag2 = all[6]
    diag2_fasit = all[8]

    eik = t.copy()
    for i in range(1,len(t)):
        eik[i] = np.real(eik2int(t[i],z,theta))
    

    inout = Fmed_inout(t,z,theta)
    inout_eik = Fmed_inout_eik(t,z,theta)
    if not fmed:
        inout=0

    full2 = theta**2/2*np.real(full2)+inout
    Nc2 = theta**2/2*np.real(Nc2)+inout
    diag2 = theta**2/2*np.real(diag2)+inout
    diag2_fasit = theta**2/2*np.real(diag2_fasit) + inout

    eik = theta**2/2*eik + inout_eik

    error = np.abs(np.real(diag2)-np.real(diag2_fasit))

    if largeNc:
        return t,np.real(full2),np.real(Nc2), np.real(diag2),error
    else:
        return t,np.real(full2),np.real(diag2),error


def theta_plot(time_frac,z_n,fmed=True):
    time_n = int(len(t)*time_frac)-1
    time = t[time_n]

    z = z_values[z_n]

    full_theta = theta_values.copy()
    Nc_theta = theta_values.copy()
    diag_theta = theta_values.copy()
    diag_fasit_theta = theta_values.copy()
    eik = theta_values.copy()

    for i,theta in enumerate(theta_values):

        all = np.load(f'{filename}_theta={theta:.3f}_z={z:.3f}_L={tmax}_E={EGev}_gridpoints={N}_gridsize={grid_size}.npy')
        full2 = all[2]
        Nc2 = all[4]
        diag2 = all[6]
        diag2_fasit = all[8]

        full_theta[i] = np.real(full2[time_n])
        Nc_theta[i] = np.real(Nc2[time_n])
        diag_theta[i] = np.real(diag2[time_n])
        diag_fasit_theta[i] = np.real(diag2_fasit[time_n])

        eik[i] = np.real(eik2int(time,z,theta))
    
        inout = Fmed_inout(time,z,theta)
        inout_eik = Fmed_inout_eik(time,z,theta)
        
        if not fmed:
            inout=0
            inout_eik=0
        
        full_theta[i] = theta**2/2*full_theta[i]+inout
        Nc_theta[i] = theta**2/2*Nc_theta[i]+inout
        diag_theta[i] = theta**2/2*diag_theta[i]+inout
        diag_fasit_theta[i] = theta**2/2*diag_fasit_theta[i]+inout

        eik[i] = theta**2/2*eik[i] + inout_eik


    error = np.abs(np.real(diag_theta)-np.real(diag_fasit_theta))

    return np.real(full_theta),np.real(diag_theta),error

def z_plot(time_frac,theta_n):
    time_n = int(len(t)*time_frac)-1
    time = t[time_n]

    theta = theta_values[theta_n]

    full_z = np.array([])
    Nc_z = np.array([])
    diag_z = np.array([])
    diag_fasit_z = np.array([])
    eik = z_values.copy()

    for i,z in enumerate(z_values):
        all = np.load(f'{filename}_theta={theta:.3f}_z={z:.3f}_L={tmax}_E={EGev}_gridpoints={N}_gridsize={grid_size}.npy')
        full2 = all[2]
        Nc2 = all[4]
        diag2 = all[6]
        diag2_fasit = all[8]

        full_z = np.append(full_z,np.real(full2[time_n]))
        Nc_z = np.append(Nc_z,np.real(Nc2[time_n]))
        diag_z = np.append(diag_z,np.real(diag2[time_n]))
        diag_fasit_z = np.append(diag_fasit_z,np.real(diag2_fasit[time_n]))

        eik[i] = np.real(eik2int(time,z,theta))

    inout = Fmed_inout(time,z_values,theta)
    inout_eik = Fmed_inout_eik(time,z_values,theta)

    full_z = theta**2/2*full_z+inout
    Nc_z = theta**2/2*Nc_z+inout
    diag_z = theta**2/2*np.real(diag_z)+inout
    diag_fasit_z = theta**2/2*diag_fasit_z + inout

    eik = theta**2/2*eik + inout_eik


    error = np.abs(np.real(diag_z)-np.real(diag_fasit_z))

    return full_z, diag_z, error




def pt_plot(time_frac,z_n,Fmed=True):
    time_n = int(len(t)*time_frac)-1
    time = t[time_n]

    z = z_values[z_n]
    
    p_values = theta_values*w(z)
    #Qs = np.sqrt(q*time)
    #p_values = p_values/Qs

    full_theta = theta_values.copy()
    Nc_theta = theta_values.copy()
    diag_theta = theta_values.copy()
    diag_fasit_theta = theta_values.copy()
    eik = theta_values.copy()

    for i,theta in enumerate(theta_values):

        all = np.load(f'{filename}_theta={theta:.3f}_z={z:.3f}_L={tmax}_E={EGev}_gridpoints={N}_gridsize={grid_size}.npy')
        full2 = all[2]
        Nc2 = all[4]
        diag2 = all[6]
        diag2_fasit = all[8]

        full_theta[i] = np.real(full2[time_n])
        Nc_theta[i] = np.real(Nc2[time_n])
        diag_theta[i] = np.real(diag2[time_n])
        diag_fasit_theta[i] = np.real(diag2_fasit[time_n])

        eik[i] = np.real(eik2int(time,z,theta))
    
        if Fmed:
            inout = Fmed_inout(z,theta,time)
            inout_eik = Fmed_inout_eik(z,theta,time)
            
            full_theta[i] = theta**2/2*full_theta[i]+inout
            Nc_theta[i] = theta**2/2*Nc_theta[i]+inout
            diag_theta[i] = theta**2/2*diag_theta[i]+inout
            diag_fasit_theta[i] = theta**2/2*diag_fasit_theta[i]+inout

            eik[i] = theta**2/2*eik[i] + inout_eik


    error = np.abs(np.real(diag_theta)-np.real(diag_fasit_theta))


    plt.rcParams['text.usetex'] = True
    #plt.text(0.01,np.max(full_theta+error),r'$\hat{q}=%.1f\,\mathrm{GeV}^2/\mathrm{fm},\, L = %.2f\,\mathrm{fm},\,z=%.2f,\,E = %.0f\,\mathrm{GeV}$'% (qhatmix,time,z,EGev),fontsize = 12)


    plt.semilogx(p_values,full_theta, label=r'Numeric $N_c=3$')
    plt.semilogx(p_values,Nc_theta, label=r'Numeric large-$N_c$')
    #plt.plot(theta_values,diag_theta, label='K2 Numeric-diag-re')
    plt.semilogx(p_values,diag_fasit_theta, label=r'Analytic large-$N_c$ hom')

    plt.fill_between(p_values, full_theta-error, full_theta+error, alpha=.3)

    plt.semilogx(p_values,eik,label=r'Analytic large-$N_c$ and eikonal hom')


    plt.xlabel(r'$p_\perp/Q_s$',fontsize =12)
    if Fmed:
        plt.ylabel(r'$F_{\mathrm{med}}$',fontsize =12)
    else:
        plt.ylabel(r'$F(\mathbf{p})$',fontsize =12)
    plt.title(r'$\hat{q}=%.1f\,\mathrm{GeV}^2/\mathrm{fm},\, L = %.2f\,\mathrm{fm},\,z=%.2f,\,E = %.0f\,\mathrm{GeV}$'% (qhatmix,time,z,EGev),fontsize=14)


    plt.legend(loc="upper left")

    if Fmed:
        plt.savefig(f'plots/Fmed_pt_{EGev}.png', bbox_inches='tight')
    else:
        plt.savefig(f'plots/Fp_pt_{EGev}.png', bbox_inches='tight')


    plt.show()



def z_plot_pt(time_frac,theta_n,pl_full=True,pl_Nc=True,pl_diag=True,pl_eik=True):
    time_n = int(len(t)*time_frac)-1
    time = t[time_n]

    theta = theta_values[theta_n]

    full_z = np.array([])
    Nc_z = np.array([])
    diag_z = np.array([])
    diag_fasit_z = np.array([])
    eik = z_values.copy()

    for i,z in enumerate(z_values):
        all = np.load(f'{filename}_theta={theta:.3f}_z={z:.3f}_L={tmax}_E={EGev}_gridpoints={N}_gridsize={grid_size}.npy')
        full2 = all[2]
        Nc2 = all[4]
        diag2 = all[6]
        diag2_fasit = all[8]

        full_z = np.append(full_z,np.real(full2[time_n]))
        Nc_z = np.append(Nc_z,np.real(Nc2[time_n]))
        diag_z = np.append(diag_z,np.real(diag2[time_n]))
        diag_fasit_z = np.append(diag_fasit_z,np.real(diag2_fasit[time_n]))

        eik[i] = np.real(eik2int(time,z,theta))

    inout = Fmed_inout(time,z_values,theta)
    inout_eik = Fmed_inout_eik(time,z_values,theta)

    full_z = theta**2/2*full_z+inout
    Nc_z = theta**2/2*Nc_z+inout
    diag_z = theta**2/2*np.real(diag_z)+inout
    diag_fasit_z = theta**2/2*diag_fasit_z + inout

    eik = theta**2/2*eik + inout_eik


    error = np.abs(np.real(diag_z)-np.real(diag_fasit_z))

    plt.rcParams['text.usetex'] = True
    #plt.text(0.01,np.max(full_z+error),r'$\hat{q}=%.1f\,\mathrm{GeV}^2/\mathrm{fm},\, L = %.2f\,\mathrm{fm},\,\theta=%.2f,\,E = %.0f\,\mathrm{GeV}$'% (qhatmix,time,theta,EGev),fontsize = 12)
    if pl_full:
        plt.semilogx(z_values,full_z, label=r'Numeric $N_c=3$')
        plt.fill_between(z_values, full_z-error, full_z+error, alpha=.3)
    if pl_Nc:
        plt.semilogx(z_values,Nc_z, label=r'Numeric large-$N_c$')
        plt.fill_between(z_values, Nc_z-error, Nc_z+error, alpha=.3)
    #plt.loglog(z_values,diag_z, label='K2 Numeric-diag-re')

    if pl_diag:
        plt.semilogx(z_values_big,Fmed_diag(time,z_values_big,theta),label=r'Analytic large-$N_c$ hom')
    if pl_eik:
        plt.semilogx(z_values_big,Fmed_diag_eik(time,z_values_big,theta),label=r'Analytic large-$N_c$ and eikonal hom')
 

    plt.xlabel(r'$z$',fontsize =12)

    plt.ylabel(r'$F_{\mathrm{med}}$',fontsize =12)
    plt.title(r'$\hat{q}=%.1f\,\mathrm{GeV}^2/\mathrm{fm},\, L = %.2f\,\mathrm{fm},\,\theta=%.2f,\,E = %.0f\,\mathrm{GeV}$'% (qhatmix,time,theta,EGev),fontsize=14)

    plt.legend(loc="upper left")


    plt.savefig(f'plots/Fmed_z_{EGev}.png', bbox_inches='tight')


    plt.show()