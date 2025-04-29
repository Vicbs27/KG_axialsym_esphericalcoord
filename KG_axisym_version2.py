#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
# import math
import scipy.special as sp
import scipy
from scipy.optimize import fsolve
from numpy.polynomial.legendre import Legendre

np.set_printoptions(precision=16)

#Parameters
N = 4
L0 = 1
# SIGMA_r = 1
A0 = 0.002
# r0 = 2
px = 2

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

# def P_colpoints(n):
#   "getting gegenbauer roots"
#   gegen_roots, _ = sp.roots_gegenbauer( 2 * n + 3 - 1,3/2)
#   gegen_col_prel = np.sort(gegen_roots)
#   return - gegen_col_prel[:n + 1]

# x_col = P_colpoints(px)


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


# In[31]:


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


# In[32]:


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


#r --> tile, x-->repelem     ###transposta inverte###
#np.kron(tile,repelem)     ###transposta inverte###

L = 5
# total bases
# Psi    = repelem(SB_.T, (px+1, px+1))  * np.tile(P_.T, (N+1, N+1))
# rPsi   = repelem(SB_r.T, (px+1, px+1)) * np.tile(P_.T, (N+1, N+1))/L0
# rrPsi  = repelem(SB_rr, (px+1, px+1)) * np.tile(P_, (N+1, N+1))
# xPsi   = repelem(SB_, (px+1, px+1))  * np.tile(P_x, (N+1, N+1))
# xxPsi  = repelem(SB_, (px+1, px+1))  * np.tile(P_xx, (N+1, N+1))

Psi    = np.kron(SB_.T,P_.T)
rPsi   = np.kron(SB_r.T,P_.T)  #/L
rrPsi  = np.kron(SB_rr.T,P_.T)  #/L**2
xPsi   = np.kron(SB_.T,P_x.T)
xxPsi  = np.kron(SB_.T,P_xx.T)

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


# In[44]:


#initial data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rt, Xt = np.meshgrid(r_col, x_col)
# R = np.tile(r_col,(len(x_col)))
# X = np.repeat(x_col,(len(r_col)))

#Definir os intervalos
r_plot = np.linspace(0, 5, 100) 
x_plot = np.linspace(-0.5, 0.5, 100)

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


def Psi_plot(n, r, x):
    list = [SB(2 * i, r) * P(2 * j, x)
            for i in range(N + 1) for j in range(px + 1)]
    return list[n]

def gaussian_approx(r,x,b0):
    res = sum(b0[k] * Psi_plot(k,r,x) for k in range((N+1) * (px+1)))
    return res

# data = np.loadtxt('resultados_a0.txt')
# data01 = data[:,1:]
# print(data01.shape)

# def gaussian_r(r,b0):
#     res = sum(b0[k] * SB(2 * k,r) for k in range((N+1)))
#     return res
              


# for i in range(0,100):
#     data_i = data[i,1:]
#     T = gaussian_approx(Rplot,Xplot,data_i)
    
    
# D = gaussian_approx(Rplot,Xplot,data01)    
# print(D.shape)
# print(T)
# np.savetxt('T', T, fmt='%.20f')

Z = gaussian_exact(A0,Rplot,Xplot)
Y = gaussian_approx(Rplot,Xplot,a0)
# print(Y)
# np.savetxt('gaussian_exact', Z, fmt='%.20f')
# np.savetxt('gaussian_approx', Y, fmt='%.20f')
W = Z - Y
# print(W)


#### PLOTS #####

# fig1 = plt.figure()
# ax1 = plt.axes(projection = '3d')
# ax1.plot_surface(Rplot, Xplot, Z, cmap='viridis')
# ax1.set_xlabel('r')
# ax1.set_ylabel('x')
# ax1.set_zlabel('phi0_exact')
# # plt.savefig('exact_phi_trunc121.png', dpi=300, bbox_inches='tight')

# fig2 = plt.figure()
# ax2 = plt.axes(projection = '3d')
# ax2.plot_surface(Rplot, Xplot, Y, cmap='viridis')
# ax2.set_xlabel('r')
# ax2.set_ylabel('x')
# ax2.set_zlabel('phi0_approx')
# # plt.savefig('approx_phi_trunc121.png', dpi=300, bbox_inches='tight')

# fig3 = plt.figure()
# ax3 = plt.axes(projection = '3d')
# ax3.plot_surface(Rplot, Xplot, W, cmap='viridis')
# ax3.set_xlabel('r')
# ax3.set_ylabel('x')
# ax3.set_zlabel('erro')
# # plt.savefig('exact_phi_trunc121.png', dpi=300, bbox_inches='tight')

# plt.figure(figsize=(6, 6))
# # fig3 = plt.figure()
# # ax3 = plt.axes(projection = '3d')
# # ax3.plot_surface()
# plt.show()


# In[45]:


# r = np.tile(r_col.reshape(px+1, 1), (N+1, px+1*N+1))
# x = np.tile(x_col.reshape(1, N+1), (px+1*N+1, px+1))

r = np.tile(r_col.repeat(px + 1).reshape(-1, 1), (1, (N + 1) * (px + 1)))
x = np.tile(np.tile(x_col, N + 1).reshape(-1, 1), (1, (N + 1) * (px + 1)))
# print('r=',r.shape)
# print('x=',x.shape)

da = np.zeros((N+1)*(px+1))
# print(da.shape)

h = 0.01
tf = 1

It = int(tf/h)
# print(It)

t = np.linspace(0, tf, It)


with open('resultados_a0.txt', 'w') as a0_file, \
     open('resultados_da.txt', 'w') as da_file:

    for i in range(It):  # Runge Kutta 4th order

        phi = np.dot(a0, Psi)
        dda = np.dot(np.dot(a0, rrPsi + 2/r*rPsi + (-x**2 + 1)*xxPsi/r**2 - 2*x*xPsi), inv_psi)
        L1 = h*(da)
        K1 = h*(dda)

        phi = np.dot(a0 + L1/2, Psi)
        dda = np.dot(np.dot(a0 + L1/2, rrPsi + 2/r*rPsi + (-x**2 + 1)*xxPsi/r**2 - 2*x*xPsi), inv_psi)
        L2 = h*(da + K1/2)
        K2 = h*(dda)

        phi = np.dot(a0 + L2/2, Psi)
        dda = np.dot(np.dot(a0 + L2/2, rrPsi + 2/r*rPsi + (-x**2 + 1)*xxPsi/r**2 - 2*x*xPsi), inv_psi)
        L3 = h*(da + K2/2)
        K3 = h*(dda)

        phi = np.dot(a0 + L3, Psi)
        dda = np.dot(np.dot(a0 + L3, rrPsi + 2/r*rPsi + (-x**2 + 1)*xxPsi/r**2 - 2*x*xPsi), inv_psi)
        L4 = h*(da + K3)
        K4 = h*(dda)

        da = da + 1/6 * (K1 + 2*K2 + 2*K3 + K4)
        a0 = a0 + 1/6 * (L1 + 2*L2 + 2*L3 + L4)
        
        a0_line = f"{t[i]:.6f} " + " ".join([f"{val:.15f}" for val in a0.flatten()])
        da_line = f"{t[i]:.6f} " + " ".join([f"{val:.15f}" for val in da.flatten()])
        
        a0_file.write(a0_line + "\n")
        da_file.write(da_line + "\n")


# In[49]:


data = np.loadtxt('resultados_a0.txt')
data01 = data[99,1:]
# print(data01)
 
def gaussian_approx(r,x,b0):
    res = sum(b0[k] * Psi_plot(k,r,x) for k in range((N+1) * (px+1)))
    return res

phi = gaussian_approx(Rplot,Xplot,data01)

fig3 = plt.figure()
ax3 = plt.axes(projection = '3d')
ax3.plot_surface(Rplot, Xplot, phi, cmap='viridis')
ax3.set_xlabel('r')
ax3.set_ylabel('x')
ax3.set_zlabel('phi')
# plt.savefig('exact_phi_trunc121.png', dpi=300, bbox_inches='tight')


# In[47]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'  #Ajuste o caminho se necessário
get_ipython().run_line_magic('matplotlib', 'notebook')

data = np.loadtxt('resultados_a0.txt')
times = data[:, 0]  #time column
all_a0 = data[:, 1:]  #a0 
r_min, r_max = 0.1, 5.0
x_min, x_max = -1, 1


#grid
r_vals = np.linspace(r_min, r_max, 30)
x_vals = np.linspace(x_min, x_max, 30)
R, X = np.meshgrid(r_vals, x_vals)

#psi para plot 3d
total_coeffs = (N+1)*(px+1)
Psi_grid = np.zeros((total_coeffs, *R.shape))

for k in range(total_coeffs):
    # Vectorize manualmente para cada k
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            Psi_grid[k,i,j] = Psi_plot(k, R[i,j], X[i,j])

#phi
phi_evolution = np.zeros((len(times), *R.shape))

for t_idx in range(len(times)):
    #Produto vetorizado: a0[t_idx] • Psi_grid
    phi_evolution[t_idx] = np.sum(all_a0[t_idx][:,None,None] * Psi_grid, axis=0)


#3D animation
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('r')
ax.set_ylabel('x')
ax.set_zlabel('ϕ(r,x,t)')
ax.set_zlim(np.min(phi_evolution), np.max(phi_evolution))

#initial frame
t_idx = 0
surf = ax.plot_surface(R, X, phi_evolution[t_idx], 
                      cmap='viridis', 
                      rstride=1, cstride=1)
ax.set_title(f'Tempo = {times[t_idx]:.3f}')

#atualização da função
def update(frame):
    global surf
    
    t_idx = min(frame*5, len(times)-1)
    
    if surf:
        surf.remove()
    
    surf = ax.plot_surface(R, X, phi_evolution[t_idx], 
                         cmap='viridis',
                         rstride=1, cstride=1)
    ax.set_title(f'Tempo = {times[t_idx]:.3f}')
    return surf

#animação
ani = FuncAnimation(fig, update, 
                   frames=len(times)//5,
                   interval=200,  
                   blit=False,   
                   repeat=True)

plt.tight_layout()
plt.show()

#salvar como gif
ani.save("evolucao_phi.gif", writer='pillow', fps=13, dpi=100)


# In[ ]:


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import quad

# # Parâmetros
# # N = 300
# # L0 = 5
# # SIGMA_r = 1
# # A0 = 0.02  
# # r0 = 2

# #exact energy
# def phi0_exact(A0,r,x):
#     return A0 * np.exp(-r**2) * (1 - (np.cos(x))**2)


# def dphi_0_dr(r):
#     return -2 * (r - r0) / SIGMA_r**2 * Phi_0(r)

# def integ(r):
#     return 2 * np.pi * r**2 * dPhi_0_dr(r)**2

# integral, erro = quad(integ, 0, np.inf)

# print('exato=',integral)


# #numerical energy
# def energy(phi_set, pi_set, drphi_set, rplot, h):
#     total_energy = np.zeros(phi_set.shape[0])  

#     for i in range(phi_set.shape[0]):
#         E_kin = np.sum(pi_set[i, :]**2 * rplot**2) * (rplot[1] - rplot[0]) 
#         E_pot = np.sum(drphi_set[i, :]**2 * rplot**2) * (rplot[1] - rplot[0])
#         total_energy[i] = 2*np.pi*(E_kin + E_pot)

#     return total_energy

# print('numérico=', total_energy)

# total_energy = energy(phi_set, pi_set, drphi_set, rplot, h)

# #abs difference
# abs_difference = np.abs(total_energy - integral)
# print('dif=',abs_difference)

# plt.plot(np.linspace(0, 10, len(total_energy)), abs_difference, 'c', label="A0=0.0002")
# plt.xlabel("t")
# plt.ylabel("|ΔE|")
# plt.ylim(0, 3e-9)
# plt.title("Diferença absoluta entre energia exata e numérica (N=300 - L0=5)")
# plt.legend()
# plt.grid(True)
# plt.show()


# In[27]:


# ### teste

# def func(r,x):
#     return Psi * r + Psi/x

# t1 = [func(r_col[j], x_col[k])
#                          for j in range(N + 1) for k in range(px + 1)]

# #r_reshaped = np.tile(r_col.reshape(M, 1), (N, M*N))
# #x_reshaped = np.tile(x_col.reshape(1, N), (M*N, M))

# # r = r_col.reshape(3,1)
# # x = x_col.reshape(3,1)
# # R = np.tile(r, (3,9))
# # X = np.tile(x, (3,9))

# R = np.tile(r_col.repeat(px + 1).reshape(-1, 1), (1, (N + 1) * (px + 1)))
# X = np.tile(np.tile(x_col, N + 1).reshape(-1, 1), (1, (N + 1) * (px + 1)))

# t2 = func(R,X)

# # print('t1=', t1)
# # # np.savetxt('t_teste1', t1, fmt='%.10f')
# # print('t2=', t2)
# # np.savetxt('t_teste2', t2, fmt='%.10f')

