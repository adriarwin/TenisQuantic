# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrià Bravo Vidal)s
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation



#Crank-Nicolson en 1 Dimensió.

#Funció triadag
#range(n) n element pero comencem a contar desde 0

L_x=17
Nx=400
dx=(2*L_x)/Nx
dt=0.01
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
    return -L_x+i*dx

xvec=np.array([x(i) for i in range(Nx+1)])
print(xvec[Nx])

def psi0(x):
    p0=200j/L_x
    n=1./((2.*np.pi)**(1/4))
    return n*np.exp(-((x-7.)**2)/4)*np.exp(-p0*x)

psi0=np.array([psi0(x(i)) for i in range(Nx+1)])


def V(x):
    a=42.55/(np.sqrt(0.1*np.pi))
    b=np.exp(-(x**2/(0.1**2)))
    
    if abs(x)==16.99:
        V=100000000
    else:
        V=a*b
    return V
Vvec=np.array([(V(x(i)+0.0001))/V(0.) for i in range(Nx+1)])

def H():
    H=np.zeros((Nx+1,Nx+1),dtype=complex)
    c=dt*(1./2)
    for i in range(Nx+1):
        for j in range(Nx+1):
            if i==j:
                H[i,j]=complex(0,c*(V(x(i))+1./(dx**2)))
            elif abs(i-j)==1:
                H[i,j]=complex(0,c*(-1./(2.*dx**2)))
    return H    
   
Hamp=H()+np.eye(Nx+1,dtype=complex)
Hamm=np.eye(Nx+1,dtype=complex)-H()

def b():
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if i==j])

def ac():
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if (i-j)==1])
    

def d(psi):
    psi1=Hamm.dot(psi) 
    return psi1


#Termes independents

bvec=b()
cvec=np.append(ac(),0)
avec=np.insert(ac(),0,0)
print(len(cvec))
print(len(avec))
print(len(bvec))

def Crannk(psi):
    #Realitza un pas de cranknicolson
    dvec=d(psi)
    return np.array(tridiag(avec,bvec,cvec,dvec),dtype=complex)

#Bucle que ens guarda dins un ftixer per cada dt el seu corresponent 
#temps.

def Crannkdef(psi0,t,dt):
    Nt=np.int(t/dt)
    psi=np.zeros((Nt,len(psi0)),dtype=complex)
    tvec=np.array([0.])
    psi[0]=psi0
    for i in range(Nt-1):        
        psi[i+1]=Crannk(psi[i])
        tvec=np.append(tvec,dt*(i+1))
    
    return psi,tvec


#Cosas de norma i tal
def norma(psi):
    return [psi[i]*np.conj(psi[i]) for i in range(Nx+1)]

def trapezis(xa,xb,dx,fun):
    Nx=np.int((xb-xa)/dx)
    funsum=(fun[0]+fun[-1])*(dx/2)
    for i in range(1,Nx):
        funsum=funsum+fun[i]*dx
    return funsum

funprobab=norma(psi0)
integral=trapezis(-L_x,L_x,dx,funprobab)
print(integral)



psi=np.zeros((Nt+1,len(psi0)),dtype=complex)
probab=np.zeros((Nt+1,len(psi0)),dtype=complex)
tvec=np.zeros(Nt+1)
psi[0]=psi0
probab[0]=norma(psi0)
tvec[0]=0.
for i in range(Nt):             
    psi[i+1]=Crannk(psi[i])
    probab[i+1]=norma(psi[i+1])
    tvec[i+1]=dt*(i+1)
    
probamax=np.max(probab[0])

Vvec=Vvec*probamax*(42.55/66.779)

#Animació
    
fig, ax = plt.subplots()
ln, = plt.plot([], [],label='Psi*Psi')
ax.plot(xvec,Vvec,label='V(x)')
ax.set_title('Colision de un paquete gaussiano')

def init():
    ax.set_xlim(-17, 17)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x/L")
    ax.set_ylabel("Probabilitat(x)")
    ax.legend()
    
    return ln, 

def update(frame):        
    ln.set_data(xvec,probab[frame])
    return ln,

ani = matplotlib.animation.FuncAnimation(fig, update, frames=range(0,Nt-1),
                    interval=10 ,init_func=init, blit=True)


plt.show()


    

    
    
