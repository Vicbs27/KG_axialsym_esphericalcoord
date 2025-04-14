#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
# import math
import scipy.special as sp
from scipy.optimize import fsolve
from numpy.polynomial.legendre import Legendre

#Parameters
N = 11
L0 = 1
SIGMA_r = 1
A0 = 0.02
r0 = 2
px = 11

##COLLOCATION POINTS

#new r collocation points with linspace
k_values = np.linspace(0, 2*N + 3, 2*N + 4, dtype=np.float64)
x__col = np.cos(np.pi * k_values / (2*N + 3))
epsilon = 1e-15  # Valor pequeno para evitar divisão por zero
r_col_pre = L0 * x__col / np.sqrt(1 - x__col**2 + epsilon)
r_col = np.flip(np.array([r_col_pre[N + 2 - k] for k in range(1, N + 2)])) #ordem inversa com np.flip
print('r =', r_col) #confere
# np.savetxt('r_col.txt', r_col, fmt='%.20f')

#collocation points for x
P_prime = sp.legendre(2 * px + 3).deriv()
x_roots = fsolve(P_prime, np.cos(np.pi * (np.arange(1, 2 * px + 3) / (2 * px + 3))))
x_col_prel = np.sort(x_roots)
x_col = -x_col_prel[:px + 1]
# -np.flip(x_col_prel[:px + 1]) #ordem inversa
print('x=',x_col) #confere
# np.savetxt('x_col.txt', x_col, fmt='%.20f')


# In[13]:


######### 2n+1 #########
##BASES
#r basis
def SB(n, r):
    return np.sin((2*n+1)*np.arctan(L0/r))
# print(SB(1,2))

def rSB(n, r):
    return -np.cos((2*n+1)*np.arctan(L0/r))*(2*n+1)*L0/(r**2*(1+L0**2/r**2)) 

def rrSB(n, r):
    return (-np.sin((2*n+1)*np.arctan(L0/r))*(2*n+1)**2*L0**2/(r**4*(1+L0**2/r**2)**2)+
2*np.cos((2*n+1)*np.arctan(L0/r))*(2*n+1)*L0/(r**3*(1+L0**2/r**2))-2*np.cos((2*n+1)*np.arctan(L0/r))*(2*n+1)*L0**3/(r**5*(1+L0**2/r**2)**2))


# #x basis
def P(i, x):
    return Legendre.basis(i)(x) 
# # sp.eval_legendre(i, x)
# # sp.legendre(i)(x)
# # print(P(12,5))

def xP(i, x):
    poly = Legendre.basis(i).deriv(1)
    return poly(x)
# # sp.eval_legendre(i, x, derivative=1)
# # sp.legendre(i).deriv()(x)

def xxP(i, x):
    poly = Legendre.basis(i).deriv(2)
    return poly(x)
# # sp.eval_legendre(i, x, derivative=2)
# # sp.legendre(i).deriv().deriv()(x)


#collocation points on the bases
# r basis
SB_ = np.zeros([N+1,N+1])
SB_r = np.zeros([N+1,N+1])
SB_rr = np.zeros([N+1,N+1])

SB_0 = np.zeros([1,N+1])

for i in range(N+1):
    SB_[i,] = SB(i,r_col)
# np.savetxt('SB_in', SB_.T, fmt='%.20f') #confere

for i in range(N+1):
    SB_r[i,] = rSB(i,r_col)
# np.savetxt('SBr_in', SB_r, fmt='%.20f') #confere

for i in range(N+1):
    SB_rr[i,] = rrSB(i,r_col)
# np.savetxt('SBrr_in', SB_rr, fmt='%.20f') #confere

# for i in range(N+1):
#     SB_0[i,] = SB(i,1)
    
# print(SB_0)

#x basis
P_ = np.zeros((px + 1, px + 1))
P_x = np.zeros((px + 1, px + 1))
P_xx = np.zeros((px + 1, px + 1))

for i in range(px + 1):
    P_[i,] = P(2*i,x_col)
# np.savetxt('P_in', P_, fmt='%.20f') #confere
    
for i in range(px + 1):
    P_x[i,] = xP(2*i,x_col)
# np.savetxt('Px_in', P_x, fmt='%.20f') #confere
    
for i in range(px + 1):
    P_xx[i,] = xxP(2*i,x_col)
# np.savetxt('Pxx_in', P_xx, fmt='%.20f') #confere


# In[14]:


###### TOTAL BASIS ######

# def repelem(arr, repeats):
#     """
#     Replicates MATLAB's repelem behavior for N-dimensional arrays.
    
#     Parameters:
#         arr (numpy.ndarray): Input array.
#         repeats (tuple): Number of repetitions along each axis.
    
#     Returns:
#         numpy.ndarray: Repeated array.
#     """
#     for axis, rep in enumerate(repeats):
#         arr = np.repeat(arr, rep, axis=axis)
#     return arr

L = 5
# total bases
# Psi    = repelem(SB_.T, (px+1, px+1))  * np.tile(P_.T, (N+1, N+1))
# rPsi   = repelem(SB_r.T, (px+1, px+1)) * np.tile(P_.T, (N+1, N+1))/L0
# rrPsi  = repelem(SB_rr, (px+1, px+1)) * np.tile(P_, (N+1, N+1))
# xPsi   = repelem(SB_, (px+1, px+1))  * np.tile(P_x, (N+1, N+1))
# xxPsi  = repelem(SB_, (px+1, px+1))  * np.tile(P_xx, (N+1, N+1))

Psi    = np.kron(P_,SB_)
rPsi   = np.kron(P_,SB_r)  #/L
rrPsi  = np.kron(P_,SB_rr)  #/L**2
xPsi   = np.kron(P_x,SB_)
xxPsi  = np.kron(P_xx,SB_)

# np.savetxt('Psi_in', Psi, fmt='%.20f')
# np.savetxt('rPsi_in', rPsi, fmt='%.20f')
# np.savetxt('rrPsi_in', rrPsi, fmt='%.20f')
# np.savetxt('xPsi_in', xPsi, fmt='%.20f')
# np.savetxt('xxPsi_in', xxPsi, fmt='%.20f')

#inverse matrix
inv_psi = np.linalg.inv(Psi)
# print(inv_psi.shape)

# print(Psi)


# In[16]:


#initial data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Rt, Xt = np.meshgrid(r_col, x_col)
R = np.tile(r_col,(len(x_col)))
X = np.repeat(x_col,(len(r_col)))

#Definir os intervalos
r_plot = np.linspace(0, 10, 100) 
x_plot = np.linspace(-1, 1, 100)

Rplot, Xplot = np.meshgrid(r_plot,x_plot)

# def gaussian(r,x):
#     return A0 * np.exp(-r**2) * (-x**2 + 1)

def gaussian_exact(r,x):
    return A0 * np.exp(-r**2) * (1 - (np.cos(x))**2)

# A0 * np.exp(-r**2 -x**2)
# A0 * np.exp(-r**2) * (1 - x**2)

# Z = gaussian_exact(Rplot,Xplot)

gaussian_col = gaussian_exact(R,X)
# print(gaussian_.shape) #(16,px=pr=3)

b0 = np.dot(inv_psi,gaussian_col)
# print('a0=',a0.shape) #(231,1)
# print("a0=",a0)

# A = np.dot(Psi[1],a0)
# print(A)

# def a0(k,r,x):
#     res = sum(inv_psi[k] * gaussian_(r,x) for k in range((N+1) * (px+1)))
#     return res

def Psi_plot(n, r, x):
    list = [SB(i, r) * P(2 * j, x)
            for i in range(N + 1) for j in range(px + 1)]
    return list[n]

def gaussian_approx(a0,r,x):
    res = sum(a0[k] * Psi_plot(k,r,x) for k in range((N+1) * (px+1)))
    return res
# print('Phi_0=',gaussian_approx(a0,Rplot,Xplot))
# np.savetxt('Phi_0', gaussian_approx(a0,Rplot,Xplot), fmt='%.20f')
# print('Phi_0[1]=',gaussian_approx(a0,R,X)[1])
# print(gaussian_approx(a0,Rplot,Xplot)[1].shape)



# Phi_0 = gaussian_approx(a0,R,X)
# Phi = Phi_0.reshape(-1,1)

fig1 = plt.figure()
ax1 = plt.axes(projection = '3d')
ax1.plot_surface(Rplot, Xplot, gaussian_exact(Rplot,Xplot), cmap='viridis')
plt.savefig('exact_phi_trunc121.png', dpi=300, bbox_inches='tight')

fig2 = plt.figure()
ax2 = plt.axes(projection = '3d')
ax2.plot_surface(Rplot, Xplot, gaussian_approx(b0,Rplot,Xplot), cmap='viridis')
plt.savefig('approx_phi_trunc121.png', dpi=300, bbox_inches='tight')
plt.show()


# In[43]:


Rm = np.repeat(r_col,11)
Rr = Rm.reshape(-1,1)
r = Rr

Xm = np.repeat(x_col,21)
Xr = Xm.reshape(-1,1)
x = Xr

da = np.zeros((N+1)*(px+1))
# print(da.shape)

h = 0.1
tf = 1

It = int(tf/h)
# print(It)

t = np.linspace(0, tf, It)

# phi_set = []
# print(phi_set.shape)
# pi_set = np.zeros([10000,231])
# drphi_set = np.zeros([10000,231])
# drrphi_set = np.zeros([10000,231])
# dxphi_set = np.zeros([10000,231])
# dxxphi_set = np.zeros([10000,231])

#open files before loop
# da_file = open('da_data.txt', 'w')
# a0_file = open('a0_data.txt', 'w')

# with open('da_data.txt', 'w') as da_file, open('a0_data.txt', 'w') as a0_file:
with open('resultados_rk4.txt', 'w') as file:
    header = "t " + " ".join([f"a{i}" for i in range(len(a0))])
    file.write(header + "\n")

    for i in range(It):  # Runge Kutta 4th order

        phi = np.dot(a0.T, Psi)
        dda = np.dot(np.dot(a0.T, rrPsi + 2/r*rPsi + (-x**2 + 1)*xxPsi/r**2 - 2*x*xPsi), inv_psi)
        L1 = h*(da)
        K1 = h*(dda)

        phi = np.dot(a0.T + L1/2, Psi)
        dda = np.dot(np.dot(a0.T + L1/2, rrPsi + 2/r*rPsi + (-x**2 + 1)*xxPsi/r**2 - 2*x*xPsi), inv_psi)
        L2 = h*(da + K1/2)
        K2 = h*(dda)

        phi = np.dot(a0.T + L2/2, Psi)
        dda = np.dot(np.dot(a0.T + L2/2, rrPsi + 2/r*rPsi + (-x**2 + 1)*xxPsi/r**2 - 2*x*xPsi), inv_psi)
        L3 = h*(da + K2/2)
        K3 = h*(dda)

        phi = np.dot(a0.T + L3, Psi)
        dda = np.dot(np.dot(a0.T + L3, rrPsi + 2/r*rPsi + (-x**2 + 1)*xxPsi/r**2 - 2*x*xPsi), inv_psi)
        L4 = h*(da + K3)
        K4 = h*(dda)

        da = da + 1/6 * (K1 + 2*K2 + 2*K3 + K4)
        a0 = a0 + 1/6 * (L1 + 2*L2 + 2*L3 + L4)
        
        line = f"{t[i]:.6f} " + " ".join([f"{val:.15f}" for val in a0.flatten()])
        file.write(line + "\n")
    
#         np.savetxt(da_file, [da.flatten()], fmt='%.8f') 
#         np.savetxt(a0_file, [a0.flatten()], fmt='%.8f')

    
#close files after loop
# da_file.close()
# a0_file.close()
    
#     phi_set[i:] = np.dot(Psiplot,a0)
#     pi_set[i,:] = np.dot(da, Psiplot)
#     drphi_set[i,:] = np.dot(a0, rPsiplot)
#     drrphi_set[i,:] = np.dot(a0, rrPsiplot)
#     dxphi_set[i,:] = np.dot(a0,xPsiplot)
#     dxxphi_set[i,:] = np.dot(a0,xxPsiplot)

# np.savetxt('phi_set.txt', phi_set, fmt='%.10f', header='phi')
# np.savetxt('drphi_set.txt', drphi_set, fmt='%.10f', header='dr_phi')
# np.savetxt('drrphi_set.txt', drrphi_set, fmt='%.10f', header='drr_phi')
# np.savetxt('dxphi_set.txt', dxphi_set, fmt='%.10f', header='dx_phi')
# np.savetxt('dxxphi_set.txt', dxxphi_set, fmt='%.10f', header='dxx_phi')




