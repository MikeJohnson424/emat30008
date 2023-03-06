#%% 

import numpy as np
import matplotlib.pyplot as plt
from integrate import solve_to
from shooting import isolate_lim_cycle
from scipy.optimize import fsolve



#%%

def h(x,alpha):
    return x**3-x+alpha


def NP_continuation(func,x0=[1], alpha0=-5,step_size = 0.01, imax = 1000):

    counter = 0

    x = np.zeros((len(x0),imax+1))

    alpha = np.arange(alpha0, alpha0+imax*step_size+step_size, step_size)

    x0 = x0[0]

    sol_old = x0

    for i in range(imax+1):
        
        alpha_current = alpha[counter]

        sol_current = fsolve(lambda x: func(x,alpha_current), sol_old)

        x[:,counter] = sol_current

        counter += 1

        sol_old = sol_current

        

    return [x, alpha]

[x, alpha] = NP_continuation(h)



plt.plot(alpha,x[0])
plt.show()


# %%


def PA_continuation(func,x0,alpha0): 

    # Linear Predictor: 

    u_old = np.hstack((alpha_old, alpha_current))
    u_current = np.hstack((alpha_current, x_current))

    delta_u = u_current - u_old

    u_future = u_current + delta_u

    # Corrector 

    def g(func,u):   

        alpha,x = u
        u = np.array(u)
        dp = np.dot((u-u_future),delta_u)

        return np.hstack((func(x,alpha),x0))

    fsolve(lambda u: g(h,u),x0)

# %%
