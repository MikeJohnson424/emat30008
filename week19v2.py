#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from math import floor

#%%

def gen_diag_mat(N,entries):

    length = len(entries)
    lim = floor(length/2)
    diagonals = range(-lim,lim+1)
    k = [[] for _ in range(length)]

    for i in range(length):
        k[i] = entries[i]*np.ones(N - abs(diagonals[i]))

    #k = [np.ones(N-1),-2*np.ones(N),np.ones(N-1)]
    A_DD = diags(k,diagonals).toarray()

    return A_DD


#%%

def f_diff_solver(x_span,b_cond,N):
    
    alpha, beta = b_cond
    a,b, = x_span
    delta_x = (b-a)/N
    x = np.linspace(a,b,N+1)

    A_DD = gen_diag_mat(N-1)
    b_DD = np.hstack((alpha,np.zeros(N-3),beta))
    
    u_sol = np.linalg.solve(A_DD,-b_DD)
    u_full = np.hstack((alpha,u_sol,beta))


    class result:
        def __init__(self,u,x):
            self.u = u
            self.x = x


    return result(u_full,x)

#%%

x_span = [0,10]; b_cond = [2,3]; N = 8; a,b = x_span; alpha,beta = b_cond; delta_x = (b-a)/N;

result = f_diff_solver([0,10],[2,3],10)
u = result.u
x = result.x

def sol_no_source(x,a,b,alpha,beta):

    return ((beta-alpha))/(b-a)*(x-a)+alpha

def sol_source_1(x,a,b,alpha,beta,D=1):

    return -1/(2*D)*(x-a)*(x-b)+((beta-alpha))/(b-a)*(x-a)+alpha

#u_true = sol_source_1(x,a,b,alpha,beta)
u_true = sol_no_source(x,a,b,alpha,beta)

plt.plot(x,u)
plt.plot(x,u_true)






# %%
