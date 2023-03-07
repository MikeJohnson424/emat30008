#%% 

import numpy as np
import matplotlib.pyplot as plt
from integrate import solve_to
from shooting import isolate_lim_cycle
from scipy.optimize import fsolve

def h(x,alpha):
    return x**3-x-alpha


#%%

def NP_continuation(func,x0=[-5],alpha0=-10,imax=1000,step_size = 0.01):

    counter = 0

    x_new = x0

    x = np.zeros((len(x0),imax))
    alpha = np.zeros(imax)

    alpha_new = alpha0

    for i in range(imax):

        alpha[counter] = alpha_new

        x_old = x_new
        alpha_old = alpha_new

        x_new = fsolve(lambda x: func(x,alpha_new),x_old)
        alpha_new = alpha_old + step_size

        x[:,counter] = x_new
        

        counter += 1
        

    return [x, alpha]

[x, alpha] = NP_continuation(h)



plt.plot(alpha,x[0])
plt.show()


# %%

def g(func, u, u_pred, delta_u):
    
    x,alpha = u

    top = func(x,alpha)

    bottom = np.dot((u - u_pred),delta_u)


    return np.array([top,bottom])

def PA_continuation(func,x0=1,alpha0=-5,imax=100, initial_step_size = 0.1): 

    u = np.zeros((2,imax+1))

    alpha0 = alpha0
    x0 = fsolve(lambda x: func(x,alpha0),x0)
    u_old = np.hstack((x0,alpha0))

    u[:,0] = u_old

    alpha1 = alpha0 + initial_step_size
    x1 = fsolve(lambda x: func(x,alpha1),x0)
    u_current = np.hstack((x1,alpha1))

    counter = 1
    

    for i in range (imax):

        u[:,counter] = u_current
        
        # Linear Predictor: 

        delta_u = u_current - u_old
        u_pred = u_current + delta_u

        # Corrector:

        u_true = fsolve(lambda u: g(func,u, u_pred, delta_u),u_current)

        # Update Values

        u_old = u_current
        u_current = u_true
        
        counter +=1


    return u



u = PA_continuation(h)

plt.plot(u[1,:],u[0,:])

# %%
