# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrià Bravo Vidal)s
"""

#Funció triadag
#range(n) n element pero comencem a contar desde 0

import numpy as np
import matplotlib.pyplot as plt
L=5.
Nx=40
dx=dy=(np.float64((2*L)/Nx))
dt=np.float64(0.1)
t=20
Nt=int(t/dt)

print(dx)


def tridiag(a, b, c, d):
    n = len(d)  # número de filas

    # Modifica los coeficientes de la primera fila
    cp= np.zeros(n,dtype=complex)
    dp=np.zeros(n,dtype=complex)
    cp[0]=c[0]/b[0]  # Posible división por cero
    dp[0]=d[0]/ b[0]

    for i in range(1, n):
        ptemp = b[i] - (a[i] * cp[i-1])
        cp[i]= c[i]/ptemp
        dp[i] = (d[i] - a[i] * dp[i-1])/ptemp

    # Sustitución hacia atrás
    x = np.zeros(n,dtype=complex)
    x[-1] = dp[-1] #-1 vol dir l'ultim element

    for i in range(-2, -n-1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]

    return x

def x(i):    
    return -L+i*dx
def y(i):
    return -L+i*dx

xvec=np.array([x(i) for i in range(Nx+1)])
yvec=np.array([y(i) for i in range(Nx+1)])
print(xvec[Nx])

def psi0f(x,y):
    p0=20j/L
    n=1./((2*np.pi)**(1/2))
    return n*np.exp(-((x-1.)**2+(y-1.)**2)/4)*np.exp(-p0*x)



psi0=np.array([[psi0f(xvec[i],yvec[j]) for i in range(Nx+1)]
              for j in range(Nx+1)])


def V(x,y):
    if abs(x)>=5.00 or abs(y)>=5.00:
        V=100000.
    else:
        V=0.
    
    return V

def Hx(n):
    H=np.zeros((Nx+1,Nx+1),dtype=complex)
    c=dt*(1./2)
    for i in range(Nx+1):
        for j in range(Nx+1):
            if i==j:
                H[i,j]=complex(0,c*(V(xvec[n],xvec[j])+1./(dx**2)))
            elif abs(i-j)==1:
                H[i,j]=complex(0,c*(-1./(2.*dx**2)))
    return H    

def Hy(n):
    H=np.zeros((Nx+1,Nx+1),dtype=complex)
    c=dt*(1./2)
    for i in range(Nx+1):
        for j in range(Nx+1):
            if i==j:
                H[i,j]=complex(0,c*(V(xvec[i],xvec[n])+1./(dx**2)))
            elif abs(i-j)==1:
                H[i,j]=complex(0,c*(-1./(2.*dx**2)))
    return H
    

def bx(n):
    Hamp=Hx(n)+np.eye(Nx+1,dtype=complex)
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if i==j])
def by(n):
    Hamp=Hy(n)+np.eye(Nx+1,dtype=complex)
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if i==j])

def ac():
    Hamp=Hx(0)+np.eye(Nx+1,dtype=complex)
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if (i-j)==1])
    

def dvecx(psi,i):
    Hamm=(np.eye(Nx+1,dtype=complex))-Hx(i)
    psi=np.dot(Hamm,psi)
   
    
    return psi

dvec=dvecx(psi0[0,:],0)

def dvecy(psi,j):
    Hamm=(np.eye(Nx+1,dtype=complex))-Hy(j)
    psi=np.dot(Hamm,psi)    
    
    return psi


#Pas de CrannkNicolson
def Crannk(psi):
    cvec=np.append(ac(),0)
    avec=np.insert(ac(),0,0)
    for i in range(Nx+1):
        bvec=bx(i)
        dvec=dvecx(psi[i,:],i)
        psi[i,:]=tridiag(avec,bvec,cvec,dvec)
    for j in range(Nx+1):
        bvec=by(j)
        dvec=dvecy(psi[:,j],j)
        psi[:,j]=tridiag(avec,bvec,cvec,dvec)
    
    return psi
    


def norma(psi):
    return np.array([[np.real((psi[i,j])*np.conj(psi[i,j])) for j in range(Nx+1)] 
             for i in range(Nx+1)])



def trapezis(xa,xb,ya,yb,dx,fun):
    Nx=np.int((xb-xa)/dx)
    Ny=np.int((yb-ya)/dx)    
    funsum=0.
    for i in range(Nx+1):
        funsum=funsum+(fun[i,0]+fun[i,-1])*(dx**2)/2
        for j in range(1,Ny):
            funsum=funsum+fun[i,j]*dx**2
    return funsum


psi=np.zeros((Nx+1,Nx+1,4),dtype=complex)
normavec=np.zeros((3,40),dtype=float)
tvec=np.zeros(40,dtype=float)
psi[:,:,0]=psi0
Nxvec=np.array([20,40,60])
for j in range(3):
    dx=(2*L)/Nxvec[j]
    Nx=Nxvec[j]
    xvec=np.array([x(i) for i in range(Nx+1)])
    yvec=np.array([y(i) for i in range(Nx+1)])
    psi0=np.array([[psi0f(x(n),y(m)) for n in range(Nx+1)] 
        for m in range(Nx+1)])
    normavec[j,0]=trapezis(-L,L,-L,L,dx,norma(psi0))
    for i in range(1,40):
        psi0=Crannk(psi0)         
        normavec[j,i]=trapezis(-L,L,-L,L,dx,norma(psi0))
        tvec[i]=dt*i
np.savetxt('normas.txt',normavec)
        
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(norma(psi[:,:,0]),cmap='viridis',extent=(-L,L,-L,L),
               origin={'lower'})
plt.xlabel('x')
plt.ylabel('y')

plt.title('t=0s')
plt.subplot(2,2,2)
plt.imshow(norma(psi[:,:,1]),cmap='viridis',extent=(-L,L,-L,L),
               origin={'lower'})
plt.xlabel('x')
plt.ylabel('y')
plt.title('t=1s')
plt.subplot(2,2,3)
plt.imshow(norma(psi[:,:,2]),cmap='viridis',extent=(-L,L,-L,L),
               origin={'lower'})
plt.xlabel('x')
plt.ylabel('y')
plt.title('t=2s')
plt.subplot(2,2,4)
plt.imshow(norma(psi[:,:,3]),cmap='viridis',extent=(-L,L,-L,L),
               origin={'lower'})        
plt.xlabel('x')
plt.ylabel('y')    
plt.title('t=3s')
plt.subplots_adjust(wspace=0.001,hspace=0.4)
plt.suptitle('Paquet gaussià (p0x=20/L,p0y=0,sigma**2=1)//dt=1s/dx=0.25')
cax = plt.axes([0.88, 0.1, 0.025, 0.8])
plt.colorbar(cax=cax,label='p(x)')
plt.text(10,10,'x')



plt.figure(figsize=(10,10))
plt.plot(tvec,normavec[0],label='dx=0.5')
plt.plot(tvec,normavec[1],label='dx=0.25')
plt.plot(tvec,normavec[2],label='dx=0.125')
plt.xlabel('t(s)')
plt.ylabel('Norma(t)')
plt.ylim((0.999945,0.999949))
plt.legend()
plt.suptitle('Conservació de la norma(dt=0.1s)')
