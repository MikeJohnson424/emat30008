import numpy as  np
from week19 import gen_diag_mat
from math import ceil

a= 0; b = 1; alpha= 1; beta = 0, D = 0.5
N = 10
f = lambda x: np.zeros(np.size(x))

x = np.linspace(a,b,N+1)
dx = (b-a)/N
x_int = x[1:-1]

C = 0.49

dt = C * dx**2 / D
t_final = 1
N_time = ceil(t_final/dt)


u = np.zeros((N_time+1,N-1))
u[0,:] = f(x_int)


for n in range(N_time):

    for i in range(N-1):

        if i==0:
            u[N+1,0] = n[n,0] + C * (alpha - 2 * u[n,0] + u[n,1])
        if 0 < i and i < N-2:
        else:









