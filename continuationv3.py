#%%

import numpy as np
import matplotlib.pyplot as plt
from shooting import shooting
from scipy.optimize import fsolve,root
from functions import PPM, h, h2


#%%

def h(x,t=None,parameters=[]):
    return x[0]**3-x[0]+parameters[0]

def continuation(myode,
                 x0,
                 par0,
                 vary_par,
                 step_size,
                 max_steps):
    
    u = np.zeros((len(x0)+len(par0),max_steps+1))

    x_dim = len(x0) # Dimension of the solution
    par_dim = len(par0) # How many parameters are there

    u_old = np.hstack((
        fsolve(lambda x: myode(x,None,par0),x0),
        par0
    ))

    par0[vary_par] += step_size

    u_current = np.hstack((
        fsolve(lambda x: myode(x,None,par0),u_old[:-par_dim]),
        par0
    ))

    u[:,0] = u_old
    counter = 1

    for _ in range(max_steps):

        u[:,counter] = u_current

        # Linear Predictor

        delta_u = u_current - u_old
        u_pred = u_current + delta_u

        # Corrector

        u_true = fsolve(lambda u: np.hstack((
            myode(u[:-par_dim],None,u[-par_dim:]),
            np.dot(u - u_pred,delta_u)
        )), u_pred)

        # Update values

        u_old = u_current
        u_current = u_true
        counter += 1        

    return u

    

#%% 

# Test with h

y = np.linspace(-1.5,1.5,100)
plt.plot(y-y**3,y)

u = continuation(h,x0 = [1],par0 = [-2],
                   vary_par = 0,
                   step_size = 0.1,
                   max_steps = 50)

plt.plot(u[-1],u[0])

# %%

myode = h; x0 = [1]; par0 = [-2]; vary_par = 0; step_size = 0.1; max_steps = 40

u = np.zeros((len(x0)+len(par0),max_steps+1))

x_dim = len(x0) # Dimension of the solution
par_dim = len(par0) # How many parameters are there

u_old = np.hstack((
    fsolve(lambda x: myode(x,None,par0),x0),
    par0
))

par0[vary_par] += step_size

u_current = np.hstack((
    fsolve(lambda x: myode(x,None,par0),u_old[:-par_dim]),
    par0
))

u[:,0] = u_old
counter = 1
# %%
