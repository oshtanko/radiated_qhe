from __future__ import division  
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.misc import factorial as f
from scipy.special import genlaguerre as Lg
import mpmath
import os.path
from numpy.linalg import matrix_power
import matplotlib.animation as animation

#------------------------------------------------------------------------------
#save the array into file
def savedata(M,filename):
    np.save('data/'+filename+'.npy', M)
    return 0
#------------------------------------------------------------------------------
#load the array from the file
def loaddata(filename):
    M = np.load('data/'+filename+'.npy')
    return M
#------------------------------------------------------------------------------
#check if the file exists
def exist(filename):
    return os.path.isfile('data/'+filename+'.npy')
#------------------------------------------------------------------------------
    
# work function
Wb = 10
# disorder strength
W = 1
#density of impurities
rho = 1
# driving amplitude
lm = 1
# driving frequency
w=1
# perpendicular momentum
Kp = 0

#=========================================================    
#             setting HS basis
#=========================================================
# number of LL
n_LL = 2
# number of states considered within each LL
N = 98

D0 = (np.sqrt(4*N+4*n_LL))

# setting basis elements in the format [#LL,m]
basis = []
for ni in range(n_LL):
    for mi in range(-ni,N-ni):
        basis += [[ni,mi]]
basis = np.array(basis)

# momentum operator
Lz = np.zeros([len(basis),len(basis)])
for si in range(len(basis)):
    Lz[si,si] = basis[si][1]
    
# x-operator
X = np.zeros([len(basis),len(basis)])
for si in range(len(basis)):
    for sj in range(len(basis)):
        if abs(basis[si][0]-basis[sj][0])==1 and abs(basis[si][1]-basis[sj][1])==0:
            X[si,sj] = 1/np.sqrt(2)
        if abs(basis[si][0]-basis[sj][0])==0 and abs(basis[si][1]-basis[sj][1])==1:
            X[si,sj] = 1/np.sqrt(2)

# LL Hamiltonian 
H0 = np.zeros([len(basis),len(basis)])
for si in range(len(basis)):
    H0[si,si] = 0.5+basis[si][0]
    
# polarized light operators
A = np.zeros([len(basis),len(basis)])
for si in range(len(basis)):
    for sj in range(len(basis)):
        if basis[si][0] == basis[sj][0]+1 and basis[si][1] == basis[sj][1]-1:
            A[si,sj] = 1

P_0LL = np.zeros([len(basis),len(basis)])
P_1LL = np.zeros([len(basis),len(basis)])
for si in range(len(basis)):
    if basis[si][0] == 0:
        P_0LL[si,si] = 1
    if basis[si][0] == 1:
        P_1LL[si,si] = 1

# -- geometric parameters of the samples
# corbino disc radia (R1 = inner, R2 = outer)
R1 = np.sqrt(2*N)/5
R2 = np.sqrt(2*N)

# squre sample sides (a = inner, b = outer)
a = np.sqrt(2*N)/4
b = np.sqrt(2*N)/1.2

def unitary(A):
    E,T = np.linalg.eigh(A)
    expE = np.diag(np.exp(-1j*E))
    expA = np.dot(T,np.dot(expE,np.linalg.inv(T)))
    return expA

#==============================================================================
# FUNCTION #1: eigenstates of Landau problem in symmetric gauge
#==============================================================================
# n -- #LL
# m -- z-momentum
# x,y -- coordinates (can be tuple)
# Phi -- normalized flux (Phi=1 corresponds to lB=1)
def eigstate(n,m,x,y,Phi):
    return np.sqrt(f(n)/(2*np.pi*2.0**m*f(n+m)))*(x+1j*y)**m*np.exp(-Phi*(x**2+y**2)/4)*Lg(n,m+1e-3)(Phi*(x**2+y**2)/2)

#==============================================================================
# FUNCTION #2: density 
#==============================================================================
# n -- #LL
# m -- z-momentum
# x,y -- coordinates (can be tuple)
# Phi -- normalized flux (Phi=1 corresponds to lB=1)
def plot_density(n,Psi,Phi=1,res=100,ifplot=True):
    # -- plotting range --
    D = D0/np.sqrt(Phi)
    x,y = np.linspace(-D,D,res),np.linspace(-D,D,res)
    X,Y = np.meshgrid(x,y)
    Wf = np.empty(len(basis),np.ndarray)
    for si in range(len(basis)):
        n1,m1 = basis[si]
        Wf[si] = eigstate(n1,m1,X,Y,Phi)
    P = 0*X
    for ni in range(len(Psi)):
        psi_x = 0*X+0j
        for si in range(len(basis)):
            psi_x += n[ni]*Psi[ni][si]*Wf[si]
        P += np.abs(psi_x)**2
    if ifplot:
        plt.pcolor(X,Y,P)
    return X,Y,P

#==============================================================================
# FUNCTION #2:  Corbino disk Hamiltonian
#==============================================================================
# Phi -- normalized flux (Phi=1 corresponds to lB=1)

def QHE_Hamiltonian(Phi=1,res = 200,tp = "corbino"):
    #=========================================================    
    #             clean system hamiltonian
    #=========================================================
    Vb = np.zeros([len(basis),len(basis)],complex)
    if Wb!=0:
        # -- integration range --
        D = D0/np.sqrt(Phi)
        # -- coordinates --
        x,y = np.linspace(-D,D,res),np.linspace(-D,D,res)
        X,Y = np.meshgrid(x,y)
        dxdy = (x[1]-x[0])*(y[1]-y[0])
        # -- boundaries potential --
        if tp == "corbino":
            V = np.float_(X**2+Y**2<R1**2)+np.float_(X**2+Y**2>R2**2) 
        if tp == "square":
            V = 1+np.float_(np.float_(X**2>a**2)+np.float_(Y**2>a**2)==0)-np.float_(np.float_(X**2>b**2)+np.float_(Y**2>b**2)==0)
        #plt.pcolor(X,Y,V)
        for si in range(len(basis)):
            n1,m1 = basis[si]
            psi1 = eigstate(n1,m1,X,Y,Phi)
            for sj in range(len(basis)):
                n2,m2 = basis[sj]
                psi2 = eigstate(n2,m2,X,Y,Phi)
                Vb[si,sj] = Wb*np.sum(psi1.conj()*psi2*V)*dxdy
    #=========================================================    
    #             disorder potential
    #=========================================================
    Vd = np.zeros([len(basis),len(basis)],complex)
    if W!=0:
        # -- positions and charge of impurities
        if tp == "corbino":
            Nimp=int(rho*np.pi*(R2**2-R1**2))
            xd,yd = np.zeros(Nimp),np.zeros(Nimp)
            r2,phi = R1**2+np.random.rand(Nimp)*(R2**2-R1**2),2*np.pi*np.random.rand(Nimp)
            xd,yd = np.sqrt(r2)*np.cos(phi),np.sqrt(r2)*np.sin(phi)
        if tp == "square":
            Nimp=int(rho*np.pi*b**2)
            xd,yd = (2*np.random.rand(Nimp)-1)*b,(2*np.random.rand(Nimp)-1)*b
            xd,yd = xd[(xd**2>a**2)+(yd**2>a**2)],yd[(xd**2>a**2)+(yd**2>a**2)]
        Q = np.random.normal(0,1,size = len(xd))
        # disordered system hamiltonian
        for si in range(len(basis)):
            n1,m1 = basis[si]
            psi1 = eigstate(n1,m1,xd,yd,Phi)
            for sj in range(len(basis)):
                n2,m2 = basis[sj]
                psi2 = eigstate(n2,m2,xd,yd,Phi)
                Vd[si,sj] = W*np.sum(Q*psi1.conj()*psi2)
    #---------------------------------------------------------
    return H0+Vb+Vd    

def plot_QHE_Ham(H,mu,tp='corbino',Phi=1):
    D = (np.sqrt(4*N+4*n_LL))/np.sqrt(Phi)
    plt.subplot(1,2,1)
    E,Psi = np.linalg.eigh(H)
    Psi = Psi.T
    plot_density(E<mu,Psi)
#    if W!=0:
#        plt.scatter(xd,yd,s=1,c=np.sign(Q),cmap = 'seismic')
    x = np.linspace(-D,D,10000)
    if tp == "corbino":
        plt.plot(x,np.sqrt(R1**2-x**2),c='w')
        plt.plot(x,-np.sqrt(R1**2-x**2),c='w')
        plt.plot(x,np.sqrt(R1**2-x**2),c='w')
        plt.plot(x,np.sqrt(R2**2-x**2),c='w')
        plt.plot(x,-np.sqrt(R2**2-x**2),c='w')
    if tp == "square":
        plt.plot([a,a],[-a,a],c='w')
        plt.plot([-a,-a],[-a,a],c='w')
        plt.plot([-a,a],[a,a],c='w')
        plt.plot([-a,a],[-a,-a],c='w')
        plt.plot([b,b],[-b,b],c='w')
        plt.plot([-b,-b],[-b,b],c='w')
        plt.plot([-b,b],[b,b],c='w')
        plt.plot([-b,b],[-b,-b],c='w')
    plt.subplot(1,2,2)
    Lz = np.zeros(len(basis))
    for si in range(len(basis)):
        Lz[si] = basis[si][1]
    M = np.zeros(len(basis))
    for si in range(len(basis)):
        M[si] = np.sum(np.abs(Psi[si])**2*np.diag(Lz))
    plt.subplot(1,2,2)
    plt.scatter(M,E)
    plt.plot([-n_LL,N-n_LL],[mu,mu],ls="--",c='k')
    for ni in range(n_LL):
        plt.plot([-n_LL,N-n_LL],[0.5+ni,0.5+ni],ls=":",c='k')
    plt.ylim(0,n_LL+1)
    return 0

#==============================================================================
# FUNCTION #2:  generate set of Hs
#==============================================================================

if W==0:
    samples = 1
if W>0:
    samples = 1
filename = "qhe_disordered_corbino_nLL"+str(n_LL)+"_N"+str(N)+"_Wb"+str(Wb)+"_W"+str(W)+"_rho"+str(rho)+"_smp"+str(samples)
if not exist(filename):
    hdata = np.empty(samples,np.ndarray)
    for si in range(samples):
        hdata[si] = QHE_Hamiltonian(tp = "corbino")
    savedata(hdata,filename)
if exist(filename):
    hdata = loaddata(filename)
    
Hqhe = hdata[0]
#plot_QHE_Ham(H=Hqhe,mu=1.,tp='corbino')

UF = np.eye(len(Hqhe))+0j
time = np.linspace(0,2*np.pi/w,100)
dt = time[1]-time[0]
#Id = np.eye(len(basis))
#exp_ikx = unitary(Kp*X)
#B = lm*np.dot(exp_ikx,Kp*Id+A)
for ti in range(len(time)):
    H = Hqhe+lm*A*np.exp(-1j*w*time[ti])+lm*A.T.conj()*np.exp(1j*w*time[ti])
    UF = np.dot(UF,unitary(H*dt))
    
#ls,nu = [],[]
#for si in range(len(basis)):
#    for sj in range(si,len(basis)):
#        if basis[si][0]==0 and basis[sj][0]==1 and basis[si][1]==int(N/2):
#            ls += [basis[si][1]-basis[sj][1]]
#            nu += [abs(UF[si,sj])**2]
#            
#ls, nu = zip(*sorted(zip(ls, nu)))
#plt.plot(ls,nu)            


#==============================================================================
#                           animation settings
#==============================================================================

periods=100
time = np.arange(periods)

def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

ax1.set_xlabel("$x/\ell_B$")
ax1.set_ylabel("$y/\ell_B$")
ax2.set_xlabel("Time (periods)")
ax2.set_ylabel("LL populations $p_0$ ($p_1$)")
ax3.set_xlabel("Time (periods)")
ax3.set_ylabel("Expected moment $M_z$")
    
x = np.linspace(-D0,D0,10000)

#==============================================================================
#                          single particle
#==============================================================================

E,Psi = np.linalg.eigh(Hqhe)
Psi = Psi.T
i =[25]#[int(0.4*N)]#[np.random.randint(len(E))]
psi = Psi[i][0]

ims = []
p0,p1,M = np.zeros(len(time)),np.zeros(len(time)),np.zeros(len(time))
for ti in np.arange(periods):
    print ti
    X,Y,P = plot_density([1],[psi],ifplot=False)
    line1 = ax1.pcolor(X, Y, P,norm=plt.Normalize(0, 0.01),cmap = 'hot')
    line2, = ax1.plot(x,-np.sqrt(R1**2-x**2),c='w',ls="--")#plt.plot(x,np.cos(x*add/10))#
    line3, = ax1.plot(x,np.sqrt(R1**2-x**2),c='w',ls="--")
    line4, = ax1.plot(x,-np.sqrt(R2**2-x**2),c='w',ls="--")
    line5, = ax1.plot(x,np.sqrt(R2**2-x**2),c='w',ls="--")
    
    p0[ti] = np.sum(np.diag(P_0LL)*np.abs(psi)**2)
    p1[ti] = 1-p0[ti]
    line6, = ax2.plot(time[:ti],p0[:ti],c='b')
    line7, = ax2.plot(time[:ti],p1[:ti],c='r')
    
    M[ti] = np.sum(np.diag(Lz)*np.abs(psi)**2)
    line8, = ax3.plot(time[:ti],M[:ti],c='green')
    line9, = ax3.plot([time[0],time[-1]],[M[0],M[0]],c='k',ls="--")
    
    ims.append([line1,line2,line3,line4,line5,line6,line7,line8,line9])
    psi = np.dot(UF,psi)


im_ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=0,blit=True)

#To save this second animation with some metadata, use the following command:
im_ani.save('qhe2.mp4', metadata={'artist':'disordered system'})

#==============================================================================
#                           many particles
#==============================================================================

#E,T = np.linalg.eigh(Hqhe)
#n = np.diag(E>0.7)
#rho = np.dot(T,np.dot(n,np.linalg.inv(T)))
#
#ims = []
#p0,p1,M = np.zeros(len(time)),np.zeros(len(time)),np.zeros(len(time))
#for ti in np.arange(periods):
#    print ti
#    n,Psi = np.linalg.eigh(rho)
#    Psi = Psi.T
#    X,Y,P = plot_density(n,Psi,ifplot=False)
#    line1 = ax1.pcolor(X, Y, P,norm=plt.Normalize(0, 0.01))
#    line2, = ax1.plot(x,-np.sqrt(R1**2-x**2),c='w',ls="--")#plt.plot(x,np.cos(x*add/10))#
#    line3, = ax1.plot(x,np.sqrt(R1**2-x**2),c='w',ls="--")
#    line4, = ax1.plot(x,-np.sqrt(R2**2-x**2),c='w',ls="--")
#    line5, = ax1.plot(x,np.sqrt(R2**2-x**2),c='w',ls="--")
#    
##    p0[ti] = np.sum(np.diag(P_0LL)*np.abs(psi)**2)
##    p1[ti] = 1-p0[ti]
##    line6, = ax2.plot(time[:ti],p0[:ti],c='b')
##    line7, = ax2.plot(time[:ti],p1[:ti],c='r')
##    
##    M[ti] = np.sum(np.diag(Lz)*np.abs(psi)**2)
##    line8, = ax3.plot(time[:ti],M[:ti],c='green')
##    line9, = ax3.plot([time[0],time[-1]],[M[0],M[0]],c='k',ls="--")
##    
##    ims.append([line1,line2,line3,line4,line5,line6,line7,line8,line9])
#    rho = np.dot(UF,np.dot(rho,UF.T.conj()))
#
#
#im_ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=0,blit=True)
#
##To save this second animation with some metadata, use the following command:
##im_ani.save('im.mp4', metadata={'artist':'Guido'})

