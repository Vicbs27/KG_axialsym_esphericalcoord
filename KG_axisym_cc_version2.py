#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
# import math
import scipy.special as sp
from scipy.optimize import fsolve
from numpy.polynomial.legendre import Legendre

#Parameters
N = 20
L0 = 1
SIGMA_r = 1
A0 = 0.0015
r0 = 2
px = 10

##COLLOCATION POINTS

#new r collocation points with linspace
k_values = np.linspace(0, 2*N + 3, 2*N + 4, dtype=np.float64)
x__col = np.cos(np.pi * k_values / (2*N + 3))
epsilon = 1e-15  # Valor pequeno para evitar divisão por zero
r_col_pre = L0 * x__col / np.sqrt(1 - x__col**2 + epsilon)
r_col = np.array([r_col_pre[N + 2 - k] for k in range(1, N + 2)])
# # print('r =', r_col) #confere
# np.savetxt('r_col.txt', r_col, fmt='%.20f')

#collocation points for x
P_prime = sp.legendre(2 * px + 3).deriv()
x_roots = fsolve(P_prime, np.cos(np.pi * (np.arange(1, 2 * px + 3) / (2 * px + 3))))
x_col_prel = np.sort(x_roots)
x_col = -np.flip(x_col_prel[:px + 1])
# print('x=',x_col) #confere
# np.savetxt('x_col.txt', x_col, fmt='%.20f')


# In[40]:


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

for i in range(N+1):
    SB_[i,] = SB(i,r_col)
# np.savetxt('SB', SB_.T, fmt='%.20f') #confere

for i in range(N+1):
    SB_r[i,] = rSB(i,r_col)
# np.savetxt('SBr', SB_r.T, fmt='%.20f') #confere

for i in range(N+1):
    SB_rr[i,] = rrSB(i,r_col)
# np.savetxt('SBrr', SB_rr.T, fmt='%.20f') #confere

#x basis
P_ = np.zeros((px + 1, px + 1))
P_x = np.zeros((px + 1, px + 1))
P_xx = np.zeros((px + 1, px + 1))

for i in range(px + 1):
    P_[i,] = P(2*i,x_col)
# np.savetxt('P', P_.T, fmt='%.20f') #confere
    
for i in range(px + 1):
    P_x[i,] = xP(2*i,x_col)
# np.savetxt('Px', P_x.T, fmt='%.20f') #confere
    
for i in range(px + 1):
    P_xx[i,] = xxP(2*i,x_col)
# np.savetxt('Pxx', P_xx.T, fmt='%.20f') #confere


# In[29]:


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

Psi    = np.kron(SB_.T,P_.T)
rPsi   = np.kron(SB_r.T,P_.T)/L
rrPsi  = np.kron(SB_rr.T,P_.T)/L**2
xPsi   = np.kron(SB_.T,P_x.T)
xxPsi  = np.kron(SB_.T,P_xx.T)

# np.savetxt('Psi', Psi, fmt='%.20f')
# np.savetxt('rPsi', rPsi, fmt='%.20f')
# np.savetxt('rrPsi', rrPsi, fmt='%.20f')
# np.savetxt('xPsi', xPsi, fmt='%.20f')
# np.savetxt('xxPsi', xxPsi, fmt='%.20f')

#inverse matrix
inv_psi = np.linalg.inv(Psi)
print(inv_psi.shape)


# In[38]:


#initial data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# R, X = np.meshgrid(r_col, x_col)
R = np.tile(r_col,(len(x_col)))
X = np.repeat(x_col,(len(r_col)))

# Definir os intervalos
r_plot = np.linspace(0.01, 5, 50) 
x_plot = np.linspace(0.01, 3, 50)

Rplot, Xplot = np.meshgrid(r_plot,x_plot)

def gaussian(r,x):
    return np.exp(-(r)**2) * (-x**2 + 1)

Z = gaussian(Rplot,Xplot)


gaussian_ = gaussian(R,X)

# Phi0 = gaussian_.reshape(-1,1)
# print(gaussian_approx.shape) #(31,61)
# print(Phi0.shape) #(1891,1)

a0 = np.dot(inv_psi,gaussian_)
# print(a0.shape) #(1891,1)
# print("a0=",a0)

def Psi_plot(n, r, x):
    list = [SB(i, r) * P(2 * j, x)
            for i in range(N + 1) for j in range(px + 1)]
    return list[n]

def gaussian_approx(a0,r,x):
    res = sum(a0[k] * Psi_plot(k,r,x) for k in range((N+1) * (px+1)))
    return res

# Phi_0 = gaussian_approx(a0,R,X)
# Phi = Phi_0.reshape(-1,1)


# ax = plt.axes(projection = '3d')
# ax.plot_surface(Rplot, Xplot, Z, cmap='viridis')
# plt.savefig('initial_phi.png', dpi=300, bbox_inches='tight')

ax = plt.axes(projection = '3d')
ax.plot_surface(Rplot, Xplot, gaussian_approx(a0,Rplot,Xplot), cmap='viridis')
plt.savefig('initial_phi.png', dpi=300, bbox_inches='tight')
plt.show()

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(r_plot, x_plot, Z)
# fig.colorbar(surf, shrink=0.5, aspect=10)
# ax.set_xlabel('r')
# ax.set_ylabel('x')
# # ax.set_zlabel('gaussian_approx(r, x)')
# ax.set_title('Initial approx data')
# # ax.view_init(elev=30, azim=45)
# plt.tight_layout()
# plt.show()

# Phi0plot = gauss_approx(R,X,a0)

# ax = plt.axes(projection = '3d')
# ax.plot_surface(R, X, Phi0plot, cmap='viridis')
# plt.savefig('initial_phi.png', dpi=300, bbox_inches='tight')  # Pode usar .jpg, .tiff, etc.
# plt.show()


# In[12]:


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

with open('da_data.txt', 'w') as da_file, open('a0_data.txt', 'w') as a0_file:

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
    
        np.savetxt(da_file, [da.flatten()], fmt='%.8f') 
        np.savetxt(a0_file, [a0.flatten()], fmt='%.8f')
    
#close files after loop
da_file.close()
a0_file.close()
    
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

#     hist_a0.append(a0.copy())
#     hist_da.append(da.copy())
# print(hist_a0)
# print(hist_da)

# # Converte as listas para arrays NumPy
# hist_a0_array = np.array(hist_a0)
# hist_da_array = np.array(hist_da)

# Salva em arquivos .txt (cada linha é uma iteração)
# np.savetxt("hist_a0.txt", hist_a0_array.reshape(hist_a0_array.shape[0], -1), fmt='%.8f')
# np.savetxt("hist_da.txt", hist_da_array.reshape(hist_da_array.shape[0], -1), fmt='%.8f')

