
#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


 # %%



def f_diff_solver(x_span,alpha,beta,N):
   
    a,b, = x_span
    delta_x = (b-a)/N
    x = np.arange(a,b+delta_x, delta_x)
    u = np.zeros(N-1)

    def F(u):

        u[0] = (u[1]-2*u[0]+alpha)/(delta_x**2)

        for i in range(1,N-3):

            u[i] = (u[i+1]-2*u[i]+u[i-1])/(delta_x**2) 

        u[-1] = (beta-2*u[-1]+u[-2])/(delta_x**2)

        return u

    u_middle = fsolve(F,u)


    u = np.hstack((alpha,u_middle,beta))

    class result:
        def __init__(self,u,x):
            self.u = u
            self.x = x

    return result(u,x)

x_span = [0,10]
alpha = 2
beta = 3
N = 10
a,b = x_span

result = f_diff_solver(x_span,alpha,beta,N)
u = result.u
x = result.x

def sol(x,a,b,alpha,beta):

    return ((beta-alpha))/(b-a)*(x-a)+alpha

u_true = sol(x,a,b,alpha,beta)


plt.plot(x,u)
plt.plot(x,u_true)










# %%
