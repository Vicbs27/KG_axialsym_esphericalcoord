#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


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


# In[4]:


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


# In[45]:


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


def Psi_plot(n, r, x):
    list = [SB(2 * i, r) * P(2 * j, x)
            for i in range(N + 1) for j in range(px + 1)]
    return list[n]

def gaussian_approx(r,x,b0):
    res = sum(b0[k] * Psi_plot(k,r,x) for k in range((N+1) * (px+1)))
    return res

data = np.loadtxt('resultados_a0.txt')
data01 = data[:,1:]
# print(data01.shape)

# def gaussian_r(r,b0):
#     res = sum(b0[k] * SB(2 * k,r) for k in range((N+1)))
#     return res
              


# for i in range(0,100):
#     data_i = data[i,1:]
#     T = gaussian_approx(Rplot,Xplot,data_i)
    
    
D = gaussian_approx(Rplot,Xplot,data01)    
print(D.shape)
# print(T)
# np.savetxt('T', T, fmt='%.20f')

Z = gaussian_exact(A0,Rplot,Xplot)
Y = gaussian_approx(Rplot,Xplot,a0)
# print(Y)
# np.savetxt('gaussian_exact', Z, fmt='%.20f')
# np.savetxt('gaussian_approx', Y, fmt='%.20f')


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
# plt.savefig('approx_phi_trunc121.png', dpi=300, bbox_inches='tight')

# plt.figure(figsize=(5, 5))
# # fig3 = plt.figure()
# # ax3 = plt.axes(projection = '3d')
# # ax3.plot_surface()
# plt.show()


# In[6]:


# Rm = np.repeat(r_col,11)
# Rr = Rm.reshape(-1,1)
# r = Rr
# Xm = np.repeat(x_col,21)
# Xr = Xm.reshape(-1,1)
# x = Xr


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

# phi_set = []
# print(phi_set.shape)
# pi_set = np.zeros([10000,231])
# drphi_set = np.zeros([10000,231])
# drrphi_set = np.zeros([10000,231])
# dxphi_set = np.zeros([10000,231])
# dxxphi_set = np.zeros([10000,231])


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


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML

# Configuração do tamanho da figura
plt.rcParams['figure.figsize'] = [10, 7]

# Criar a figura e o eixo 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Configurações do gráfico
ax.set_title('Onda Senoide 3D Animada')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(-2, 2)

# Preparar os dados
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)

# Função para atualizar a animação
def update(frame):
    ax.clear()
    ax.set_title(f'Onda Senoide 3D - Frame {frame}')
    ax.set_xlabel('X')
    ax.set_ylim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(-2, 2)
    
    # Calcular a onda senoide com variação temporal
    Z = np.sin(np.sqrt(X**2 + Y**2) - frame/5)
    
    # Plotar a superfície
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', rstride=2, cstride=2)
    
    return surf,

# Criar a animação
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)

# Mostrar a animação no notebook
HTML(ani.to_jshtml())


# In[19]:


def pi_teste(a,b,c):

    drphi = np.dot(rPsi,c)
    drrphi = np.dot(rrPsi,c)
    dxphi = np.dot(xPsi,c)
    dxxphi = np.dot(xxPsi,c)

    RHS = 2/a * drphi + drrphi + (1 - b**2) * dxxphi/a**2 - 2*b * dxphi/a**2

    return np.dot(inv_psi,RHS)

r = np.tile(r_col.repeat(px + 1).reshape(-1, 1), (1, (N + 1) * (px + 1)))
x = np.tile(np.tile(x_col, N + 1).reshape(-1, 1), (1, (N + 1) * (px + 1)))

data = np.loadtxt('resultados_a0.txt')
data01 = data[0,1:]
# print(data01)

teste_pi = pi_teste(r,x,a0)

def phi_teste(c):
    phi_in = np.dot(Psi,c)
    res = np.dot(inv_psi,phi_in)
    
    return res

teste_phi = phi_teste(a0)


# print('a0=', teste_phi)
# print("da=",teste_pi)


# In[ ]:


# 2D Animation plot for Phi: Scalar Field dispersion

from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib as mpl


fig = plt.figure()
ax = plt.axes(xlim=(0, 10),ylim = (-1.5, 1.5))
line, = ax.plot([], [], lw=2)
initA0_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
x = rplot

def init():
    line.set_data([], [])
    initA0_text.set_text('')
    time_text.set_text('')
    return line,

def animate(i):
  y = phi_set_disp[i]
  line.set_data(x, y)
  initA0_text.set_text("$A_0 = {:}$".format(A0))
  time_text.set_text("Time ="+str(round(h+h*i,2)))
  return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=It, interval=0.05, blit=True)


anim.save("animation_KD_dispersion.mp4")


# In[32]:


data = np.loadtxt('resultados_a0.txt')
data01 = data[:,1:]
# print(data01)


# In[25]:


import numpy as np

line_data = np.loadtxt('resultados_a0.txt', skiprows=6, max_rows=1)  #skiprows sempre +1 em relação a linha
print("Line 5 numerical data:")
print(line_data)


# In[ ]:


## reconstruct phi##

"a0 tem tamanho (100,N+1 * px+1)"
"gerar um psiplot tamanho (N+1 * px+1,100) ?"


# In[37]:


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Load a0 data from file
# data = np.loadtxt('resultados_a0.txt')  # Assuming time in first column
# t = data[:, 0]
# a0 = data[:, 1:]  # All remaining columns are coefficients

# # Parameters (adjust these to match your simulation)
# # N = 10       # Number of radial modes (adjust)
# # px = 15      # Number of x modes (adjust)
# n_coeffs = a0.shape[1]  # Should equal (N+1)*(px+1)

# # Define your basis functions (example using Chebyshev and Legendre)
# # def SB(n, r):
# #     """Radial basis function (e.g., Chebyshev)"""
# #     return np.polynomial.chebyshev.Chebyshev.basis(n)(r)

# # def P(n, x):
# #     """Axial basis function (e.g., Legendre)"""
# #     return np.polynomial.legendre.Legendre.basis(n)(x)

# # Create evaluation grid
# r_pts = np.linspace(0, 1, 100)  # Radial coordinates (adjust range)
# x_pts = np.linspace(-1, 1, 100)  # Axial coordinates (adjust range)
# R, X = np.meshgrid(r_pts, x_pts)
# r_flat, x_flat = R.flatten(), X.flatten()

# # Precompute Psiplot matrix
# Psiplot = np.zeros((n_coeffs, len(r_flat)))

# for n in range(n_coeffs):
#     i = n // (px + 1)  # Radial mode index
#     j = n % (px + 1)   # Axial mode index
#     Psiplot[n, :] = SB(2*i, r_flat) * P(2*j, x_flat)  # Using your indexing scheme

# # Reconstruct field at all times
# phi = np.dot(a0, Psiplot)  # Shape: (n_times, n_points)
# phi = phi.reshape((len(t), len(x_pts), len(r_pts)))  # Reshape to spatial grid

# # Create animation
# fig, ax = plt.subplots(figsize=(10, 6))
# im = ax.imshow(phi[0].T, 
#               extent=[x_pts.min(), x_pts.max(), r_pts.min(), r_pts.max()],
#               origin='lower', aspect='auto', cmap='viridis')
# plt.colorbar(im, label='φ(r,x)')
# ax.set_xlabel('x')
# ax.set_ylabel('r')
# title = ax.set_title(f'Field Evolution at t = {t[0]:.3f}')

# def update(frame):
#     im.set_data(phi[frame].T)
#     im.set_clim(phi[frame].min(), phi[frame].max())
#     title.set_text(f'Field Evolution at t = {t[frame]:.3f}')
#     return im, title

# ani = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)
# plt.tight_layout()
# plt.show()

# # To save the animation (requires ffmpeg)
# # ani.save('phi_evolution.mp4', writer='ffmpeg', fps=30, dpi=200)


# In[27]:


### teste

def func(r,x):
    return Psi * r + Psi/x

t1 = [func(r_col[j], x_col[k])
                         for j in range(N + 1) for k in range(px + 1)]

#r_reshaped = np.tile(r_col.reshape(M, 1), (N, M*N))
#x_reshaped = np.tile(x_col.reshape(1, N), (M*N, M))

# r = r_col.reshape(3,1)
# x = x_col.reshape(3,1)
# R = np.tile(r, (3,9))
# X = np.tile(x, (3,9))

R = np.tile(r_col.repeat(px + 1).reshape(-1, 1), (1, (N + 1) * (px + 1)))
X = np.tile(np.tile(x_col, N + 1).reshape(-1, 1), (1, (N + 1) * (px + 1)))

t2 = func(R,X)

# print('t1=', t1)
# # np.savetxt('t_teste1', t1, fmt='%.10f')
# print('t2=', t2)
# np.savetxt('t_teste2', t2, fmt='%.10f')

