#%%

import numpy as np
import matplotlib.pyplot as plt
from shooting import shooting
from scipy.optimize import fsolve,root
from functions import PPM, h

#%%

def continuation(myode,          # the ode to solve
    x0,                          # the guess for the initial solution
    par0,                        # the initial parameters
    vary_par=0,                  # the index of the parameter to vary in par0
    step_size=0.1,               # the size of the initial step
    max_steps=25,                # the number of steps to take
    discretisation=None,         # the discretisation to use
    solver=fsolve):              # the solver to use

    # Define functions that will have the same inputs but output will depend on discretisation

    if discretisation == shooting:
        def func(myode, x, parameters): # Function to find limit cycle of a system
            return discretisation(myode, x, parameters)
        
    else:
        def func(myode,x,parameters): # Function to find equilibrium points of ODE (RHS set to zero)
            return myode(x,None,parameters)


    u = np.zeros((len(x0)+1,max_steps+1))

    # Produce two initial solutions to use in the predictor-corrector method

    u_old = np.hstack((solver(lambda x: func(myode,x,par0),x0),par0[vary_par]))
    par0[vary_par] += step_size # Vary chosen parameter
    u_current = np.hstack((solver(lambda x: func(myode,x,par0),x0),par0[vary_par]))
    u[:,0] = u_old 
    counter = 1

    for _ in range (max_steps):

        u[:,counter] = u_current
        
        # Linear Predictor: 

        delta_u = u_current - u_old
        u_pred = u_current + delta_u

        # Corrector:

        par0[vary_par] = u_pred[-1] # Vary chosen parameter

        def g(func, u, u_pred, delta_u):
    
            top = func(myode,u[:-1],par0)
            bottom = np.dot((u - u_pred),delta_u)

            return np.array([top,bottom])

        u_true = solver(lambda u: g(func,u, u_pred, delta_u),u_current)

        # Update Values

        u_old = u_current
        u_current = u_true
        counter +=1

    return u

y = np.linspace(-1.5,1.5,100)
plt.plot(y-y**3,y)

u_h = continuation(h,[1],[-2])
plt.plot(u_h[1],u_h[0])




# %%

myode = h
par0 = [-0.6]
x0 = [1]
discretisation = None
vary_par=0                # the index of the parameter to vary in par0
step_size=0.01              # the size of the initial step
max_steps=500               # the number of steps to take
discretisation=None         # the discretisation to use
solver=fsolve

# %%
