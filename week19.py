
#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve,root

def q(x,u,*args):

    return 1

 # %%


def f_diff_solver(x_span,alpha,beta,N):
   
    D = 1
    a,b, = x_span
    delta_x = (b-a)/N
    x = np.linspace(a,b,N+1)
    u_mid = np.zeros(N-1)

    RHS = u_mid

    def FDA(u_mid):

        RHS[0] = D*(u_mid[1]-2*u_mid[0]+alpha)/(delta_x**2) + q(x[1],u_mid[0])

        for i in range(1,N-2):

            RHS[i] = D*(u_mid[i+1]-2*u_mid[i]+u_mid[i-1])/(delta_x**2) + q(x[i+1],u_mid[i])

        RHS[-1] = D*(beta-2*u_mid[-1]+u_mid[-2])/(delta_x**2) + q(x[-2],u_mid[-1])

        return RHS 

    u_mid = fsolve(FDA,u_mid)


    u = np.hstack((alpha,u_mid,beta))

    class result:
        def __init__(self,u,x):
            self.u = u
            self.x = x

    return result(u,x)

# %%

def f_diff_solver(x_span,b_cond,N, b_con_type):

    D = 1
    a,b, = x_span
    delta_x = (b-a)/N
    x = np.linspace(a,b,N+1)
    u = np.zeros(N+1)
    

    if b_con_type == 'dirichlet': # Double Dirichlet boundary conditions

        alpha,beta = b_cond

        u[0] = alpha
        u[-1] = beta      
        u_mid = u[1:-1]

        def FDA(u_mid):

            RHS = D*(np.hstack((u_mid[1:],u[-1])) - 2*u_mid + np.hstack((u[0],u_mid[:-1])))/delta_x**2 + q(x,u_mid,2)

            return RHS


        u_mid = fsolve(FDA,u_mid)
        u = np.hstack((alpha,u_mid,beta))

    elif b_con_type == 'neumann': # Neumann boundary conditions
        u_mid = u[1:]
        alpha,delta = b_cond

        def FDA(u_mid):

            RHS1 = (u_mid[1:] - 2*u_mid[:-1] + np.hstack((alpha,u_mid[:-2])))/delta_x**2 + q(x,u_mid,2)
            RHS2 = (2*u_mid[-1]+2*u_mid[-2])/delta_x**2 + 2*delta/delta_x + q(x,u_mid,2)

            return np.hstack((RHS1,RHS2))
        
        u_mid = fsolve(FDA,u_mid)

        u = np.hstack((alpha,u_mid))   

    else: # Robin boundary conditions
        pass

    class result:
        def __init__(self,u,x):
            self.u = u
            self.x = x

    return result(u,x)

# %%

x_span = [0,10]; b_cond = [2,3]; N = 100; a,b = x_span; alpha,beta = b_cond; delta_x = (b-a)/N;

result = f_diff_solver(x_span,b_cond,N,'dirichlet')
u = result.u
x = result.x

def sol_no_source(x,a,b,alpha,beta):

    return ((beta-alpha))/(b-a)*(x-a)+alpha

def sol_source_1(x,a,b,alpha,beta,D=1):

    return -1/(2*D)*(x-a)*(x-b)+((beta-alpha))/(b-a)*(x-a)+alpha

u_true = sol_source_1(x,a,b,alpha,beta)
#u_true = sol_no_source(x,a,b,alpha,beta)

plt.plot(x,u)
plt.plot(x,u_true)


# %%
