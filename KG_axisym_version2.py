#!/usr/bin/env python
# coding: utf-8

# In[107]:


import numpy as np
import matplotlib.pyplot as plt
# import math
import scipy.special as sp
from scipy.optimize import fsolve
from numpy.polynomial.legendre import Legendre

np.set_printoptions(precision=16)

#Parameters
N = 3
L0 = 1
# SIGMA_r = 1
A0 = 0.002
# r0 = 2
px = 3

##COLLOCATION POINTS

#new r collocation points with linspace
k_values = np.linspace(0, 2*N + 3, 2*N + 4, dtype=np.float64)
x__col = np.cos(np.pi * k_values / (2*N + 3))
epsilon = 1e-15  # Valor pequeno para evitar divisão por zero
r_col_pre = L0 * x__col / np.sqrt(1 - x__col**2 + epsilon)
r_col = np.flip(np.array([r_col_pre[N + 2 - k] for k in range(1, N + 2)])) #ordem inversa com np.flip
# print('r =', r_col)#confere
# print(r_col.shape)
# np.savetxt('r_col.txt', r_col, fmt='%.20f')

#collocation points for x
P_prime = sp.legendre(2 * px + 3).deriv()
x_roots = fsolve(P_prime, np.cos(np.pi * (np.arange(1, 2 * px + 3) / (2 * px + 3))))
x_col_prel = np.sort(x_roots)
x_col = -x_col_prel[:px + 1]
# -np.flip(x_col_prel[:px + 1]) #ordem inversa
# print('x=',x_col) #confere
# print(x_col.shape)
# np.savetxt('x_col.txt', x_col, fmt='%.20f')


# """Collocation points for r"""

# def coly(k):
#     res = np.cos(np.pi * k / (2 * N + 3))
#     return res

# def colr(k):
#     res = L0 * coly(k) / (np.sqrt(1 - (coly(k)**2)))
#     return res


# """Collocation points for r shifted"""

# PR_shift = N + 1

# r_col = np.zeros(PR_shift)

# for j in range(0, PR_shift):
#     r_col[j] = colr(j + 1)

# # print(coly(1))
# # print(colr(1))
# # print(col_r_shift[0])

# # print("col_r=", r_col)

# """Collocation points for x"""

# def P_colpoints(n):
#     P_prime = sp.legendre(2 * n + 3).deriv()
#     x_roots = fsolve(P_prime, np.cos(np.pi * (np.arange(1, 2 * n + 3) / (2 * n + 3))))
#     #x_col_prel = np.sort(x_roots)
#     # x_col = np.flip(x_roots[:n + 1])
#     x__col = x_roots[:n + 1] 
     
#     return x__col

# """Collocation points for x shifted"""

# # print(P_colpoints(3 + 1))

# x_col = P_colpoints(px)

# # print("col_x=", x_col)


# In[108]:


######### 2n+1 #########
##BASES
#r basis
def SB(n, r):
    return np.sin((n+1)*np.arctan(L0/r))

# np.sin((n + 1) * ((np.pi / 2) - np.arctan(r / L0)))

# np.sin((n+1)*np.arctan(L0/r))
# print(SB(1,2))

def rSB(n, r):
    return -np.cos((n+1)*np.arctan(L0/r))*(n+1)*L0/(r**2*(1+L0**2/r**2)) 

# - np.cos((n + 1) * ((np.pi / 2) - np.arctan(r / L0))
#                    ) * (n + 1) / (L0 * (1 + (r / L0)**2))

# -np.cos((n+1)*np.arctan(L0/r))*(n+1)*L0/(r**2*(1+L0**2/r**2)) 

def rrSB(n, r):
    return (-np.sin((n+1)*np.arctan(L0/r))*(n+1)**2*L0**2/(r**4*(1+L0**2/r**2)**2)+2*np.cos((n+1)*np.arctan(L0/r))*(n+1)*L0/(r**3*(1+L0**2/r**2))-2*np.cos((n+1)*np.arctan(L0/r))*(n+1)*L0**3/(r**5*(1+L0**2/r**2)**2))

# (2 * np.cos((n + 1) * (np.pi / 2 - np.arctan(r / L0))) *
#            (n + 1) * r / (L0**3 * (1 + (r / L0)**2)**2)
#            - np.sin((n + 1) * (np.pi / 2 - np.arctan(r / L0))) *
#            (n + 1)**2 / (L0**2 * (1 + (r / L0)**2)**2)
#            )

# (-np.sin((n+1)*np.arctan(L0/r))*(n+1)**2*L0**2/(r**4*(1+L0**2/r**2)**2)+2*np.cos((n+1)*np.arctan(L0/r))*(n+1)*L0/(r**3*(1+L0**2/r**2))-2*np.cos((n+1)*np.arctan(L0/r))*(n+1)*L0**3/(r**5*(1+L0**2/r**2)**2))


# #x basis
def P(i, x):
    return sp.legendre(i)(x)

# Legendre.basis(i)(x) 
# # sp.eval_legendre(i, x)
# # sp.legendre(i)(x)
# # print(P(12,5))

def xP(i, x):
    return sp.legendre(i).deriv()(x)
# poly = Legendre.basis(i).deriv(1) returuno poly(x)
# # sp.eval_legendre(i, x, derivative=1)
# # sp.legendre(i).deriv()(x)

def xxP(i, x):
    return sp.legendre(i).deriv().deriv()(x)
# poly = Legendre.basis(i).deriv(2)return poly(x)
# # sp.eval_legendre(i, x, derivative=2)
# # sp.legendre(i).deriv().deriv()(x)


#collocation points on the bases
# r basis
SB_ = np.zeros([N+1,N+1])
SB_r = np.zeros([N+1,N+1])
SB_rr = np.zeros([N+1,N+1])

SB_0 = np.zeros([1,N+1])

for i in range(N+1):
    SB_[i,] = SB(2 * i,r_col)
# np.savetxt('SB_in', SB_.T, fmt='%.20f') #confere

for i in range(N+1):
    SB_r[i,] = rSB(2 * i,r_col)
# np.savetxt('SBr_in', SB_r, fmt='%.20f') #confere

for i in range(N+1):
    SB_rr[i,] = rrSB(2 * i,r_col)
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


# In[109]:


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


#r --> tile, x-->repelem, transposta inverte#
#np.kron(tile,repelem), transposta inverte#

L = 5
# total bases
# Psi    = repelem(SB_.T, (px+1, px+1))  * np.tile(P_.T, (N+1, N+1))
# rPsi   = repelem(SB_r.T, (px+1, px+1)) * np.tile(P_.T, (N+1, N+1))/L0
# rrPsi  = repelem(SB_rr, (px+1, px+1)) * np.tile(P_, (N+1, N+1))
# xPsi   = repelem(SB_, (px+1, px+1))  * np.tile(P_x, (N+1, N+1))
# xxPsi  = repelem(SB_, (px+1, px+1))  * np.tile(P_xx, (N+1, N+1))

Psi    = np.kron(SB_.T,P_.T)
# rPsi   = np.kron(P_,SB_r)  #/L
# rrPsi  = np.kron(P_,SB_rr)  #/L**2
# xPsi   = np.kron(P_x,SB_)
# xxPsi  = np.kron(P_xx,SB_)

# print('Psi=', Psi)

# np.savetxt('Psi_in_col', Psi, fmt='%.20f')
# np.savetxt('rPsi_in', rPsi, fmt='%.20f')
# np.savetxt('rrPsi_in', rrPsi, fmt='%.20f')
# np.savetxt('xPsi_in', xPsi, fmt='%.20f')
# np.savetxt('xxPsi_in', xxPsi, fmt='%.20f')

#inverse matrix
inv_psi = np.linalg.inv(Psi)
# print('inv_psi=', inv_psi)
# np.savetxt('inv_psi_col', inv_psi, fmt='%.20f')


# In[112]:


#initial data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rt, Xt = np.meshgrid(r_col, x_col)
# R = np.tile(r_col,(len(x_col)))
# X = np.repeat(x_col,(len(r_col)))

#Definir os intervalos
r_plot = np.linspace(0, 10, 100) 
x_plot = np.linspace(-1, 1, 100)

Rplot, Xplot = np.meshgrid(r_plot,x_plot)

# def gaussian(r,x):
#     return A0 * np.exp(-r**2) * (-x**2 + 1)

def gaussian_exact(A0,r,x):
    return A0 * np.exp(-r**2) * (1 - (np.cos(x))**2)

# A0 * np.exp(-r**2 -x**2)
# A0 * np.exp(-r**2) * (1 - x**2)


# gaussian_col = gaussian_exact(A0,R,X)
# gaussian_new = gaussian_col.reshape(-1,1)


coef_col = [gaussian_exact(A0, r_col[j], x_col[k])
                         for j in range(N + 1) for k in range(px + 1)]

a0 = np.dot(inv_psi,coef_col)
# b0 = np.einsum('jl,l->j', inv_psi, coef_col)
# print('b0=', b0) 

# alpha0 = np.einsum('jl,l->j', inv_psi, gaussian_col)
# print('alpha0=', alpha0)


# def a0(k,r,x):
#     res = sum(inv_psi[k] * gaussian_(r,x) for k in range((N+1) * (px+1)))
#     return res

def Psi_plot(n, r, x):
    list = [SB(2 * i, r) * P(2 * j, x)
            for i in range(N + 1) for j in range(px + 1)]
    return list[n]

def gaussian_approx(r,x,b0):
    res = sum(b0[k] * Psi_plot(k,r,x) for k in range((N+1) * (px+1)))
    return res



Z = gaussian_exact(A0,Rplot,Xplot)
Y = gaussian_approx(Rplot,Xplot,a0)
# np.savetxt('gaussian_exact', Z, fmt='%.20f')
# np.savetxt('gaussian_approx', Y, fmt='%.20f')


#### PLOTS #####

fig1 = plt.figure()
ax1 = plt.axes(projection = '3d')
ax1.plot_surface(Rplot, Xplot, Z, cmap='viridis')
ax1.set_xlabel('r')
ax1.set_ylabel('x')
ax1.set_zlabel('phi0_exact')
# plt.savefig('exact_phi_trunc121.png', dpi=300, bbox_inches='tight')

fig2 = plt.figure()
ax2 = plt.axes(projection = '3d')
ax2.plot_surface(Rplot, Xplot, Y, cmap='viridis')
ax2.set_xlabel('r')
ax2.set_ylabel('x')
ax2.set_zlabel('phi0_approx')
# plt.savefig('approx_phi_trunc121.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(5, 5))
# fig3 = plt.figure()
# ax3 = plt.axes(projection = '3d')
# ax3.plot_surface()
plt.show()


# In[46]:


# Rm = np.repeat(r_col,11)
# Rr = Rm.reshape(-1,1)
# r = Rr

# Xm = np.repeat(x_col,21)
# Xr = Xm.reshape(-1,1)
# x = Xr

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




