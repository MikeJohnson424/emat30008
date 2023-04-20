#%%

import numpy as np
import matplotlib.pyplot as plt
from shooting import shooting
from scipy.optimize import fsolve,root
from functions import PPM, h

def continuation(myode,  # the ODE to use
    x0,  # the initial state
    par0,  # the initial parameters
    vary_par=0,  # the index of the parameter to vary
    step_size=0.1,  # the size of the steps to take
    max_steps=100,  # the number of steps to take
    discretisation=shooting,  # the discretisation to use
    solver=root):  # the solver to use
    
    u = np.zeros((len(x0)+len(par0),max_steps+1)) # Initialise matrix to contain solution and parameters

    x_dim = len(x0) # Dimension of the solution
    par_dim = len(par0) # How many parameters are there

    u_old = np.hstack((
        solver(lambda x: myode(x,None,par0),x0).x,
        par0
    ))

    par0[vary_par] += step_size

    u_current = np.hstack((
        solver(lambda x: myode(x,None,par0),u_old[:-par_dim]).x,
        par0
    ))

    u[:,0] = u_old # Store first solution

    for i in range(max_steps):

        u[:,i+1] = u_current

        # Linear Predictor

        delta_u = u_current - u_old
        u_pred = u_current + delta_u

        # Corrector


        u_true = solver(lambda u: np.hstack((
            myode(u[:-par_dim],None,u[-par_dim:]),
            np.dot(u - u_pred,delta_u)
        )), u_pred).x

        # Update values

        u_old = u_current
        u_current = u_true      

    return u
  

#%% 

""" TEST WITH h """

y = np.linspace(-1.5,1.5,100)
plt.plot(y-y**3,y)

u = continuation(h,x0 = [1],par0 = [-2],
                   vary_par = 0,
                   step_size = 0.1,
                   max_steps = 50)
plt.plot(u[-1],u[0])

# %%

""" TEST WITH PPM """

u = continuation(PPM,x0 = [1,0],par0 = [0.5,0.2,0.1],vary_par=1)


#%%

myode = PPM; x0 = [5,6]; par0 = [0.5,0.2,0.1]; vary_par = 0; step_size = 0.1; max_steps = 40; solver = fsolve

u = np.zeros((len(x0)+len(par0),max_steps+1)) # Initialise matrix to contain solution and parameters

x_dim = len(x0) # Dimension of the solution
par_dim = len(par0) # How many parameters are there

u_old = np.hstack((
        solver(lambda x: myode(x,None,par0),x0),
        par0
    ))

par0[vary_par] += step_size

u_current = np.hstack((
    solver(lambda x: myode(x,None,par0),u_old[:-par_dim]),
    par0
))
counter = 1

# %%

u[:,counter] = u_current

# Linear Predictor

delta_u = u_current - u_old
u_pred = u_current + delta_u

# Corrector

u_true = solver(lambda u: np.hstack((
    myode(u[:-par_dim],None,u[-par_dim:]),
    np.dot(u - u_pred,delta_u)
)), u_pred)

# Update values

u_old = u_current
u_current = u_true   
counter += 1
# %%
