
#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


 # %%

def f_diff_solver(x_span,alpha,beta,N):
   
    a,b, = x_span
    delta_x = (b-a)/N
    x = np.linspace(a,b,N+1)
    u_mid = np.zeros(N-1)

    RHS = u_mid

    def FDA(u_mid):

        RHS[0] = (u_mid[1]-2*u_mid[0]+alpha)/(delta_x**2)

        for i in range(1,N-2):

            RHS[i] = (u_mid[i+1]-2*u_mid[i]+u_mid[i-1])/(delta_x**2) 

        RHS[-1] = (beta-2*u_mid[-1]+u_mid[-2])/(delta_x**2)

        return RHS 

    u_mid = fsolve(FDA,u_mid)


    u = np.hstack((alpha,u_mid,beta))

    class result:
        def __init__(self,u,x):
            self.u = u
            self.x = x

    return result(u,x)



#%%

x_span = [0,97]; alpha = 2; beta = 3; N = 11; a,b = x_span
result = f_diff_solver(x_span,alpha,beta,N)
u = result.u
x = result.x

def sol(x,a,b,alpha,beta):

    return ((beta-alpha))/(b-a)*(x-a)+alpha

u_true = sol(x,a,b,alpha,beta)


plt.plot(x,u)
plt.plot(x,u_true)










# %%
