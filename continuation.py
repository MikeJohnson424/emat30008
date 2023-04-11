#%% 

import numpy as np
import matplotlib.pyplot as plt
from shooting import shooting
from scipy.optimize import fsolve,root
from functions import PPM, h

#%%

def NP_continuation(func,x0=[-5],alpha0=-10,imax=1000,step_size = 0.01):

    counter = 0 # Counter for the number of iterations

    x_new = x0

    x = np.zeros((len(x0),imax))
    alpha = np.zeros(imax)

    alpha_new = alpha0

    for i in range(imax):

        alpha[counter] = alpha_new # Store the value of alpha

        x_old = x_new 
        alpha_old = alpha_new

        x_new = fsolve(lambda x: func(x,alpha_new),x_old) # Solve for x
        alpha_new = alpha_old + step_size # Update alpha

        x[:,counter] = x_new # Store the value of x
    
        counter += 1
        
    return [x, alpha]


# %%
""" == WORKING PSUEDO ARCLENGTH CONTINUATION FOR h(x,alpha) == """

def g(func, u, u_pred, delta_u):
    
    x,alpha = u

    top = func(x,alpha)
    bottom = np.dot((u - u_pred),delta_u)

    return np.array([top,bottom])

def PA_continuation(func,x0,alpha0=[-5],imax=100, initial_step_size = 0.1): 

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

# %%

""" == PA CONTINUATION FOR ODEs == """

def continuation(myode,  # the ODE to use
    x0,  # the initial state
    par0,  # the initial parameters
    vary_par=0,  # the parameter to vary
    step_size=0.1,  # the size of the steps to take
    max_steps=10,  # the number of steps to take
    discretisation=shooting,  # the discretisation to use
    solver=fsolve):  # the solver to use

    # Initialise the arrays to store the results

    u = np.zeros((len(x0)+1,max_steps+1))

    # Set the initial values

    u_old = np.hstack((solver(lambda x: discretisation(myode,x,par0),x0),par0[vary_par]))
    par0[vary_par] += step_size
    u_current = np.hstack((solver(lambda x: discretisation(myode,x,par0),x0),par0[vary_par]))
    u[:,0] = u_old
    counter = 1

    for i in range (max_steps):

        u[:,counter] = u_current
        
        # Linear Predictor: 

        delta_u = u_current - u_old
        u_pred = u_current + delta_u

        # Corrector:

        par0[vary_par] = u_pred[-1] # Update parameters
        u_true = solver(lambda u: np.hstack((discretisation(myode, u[:-1], par0),
                                            np.dot((u - u_pred),delta_u))),
                                            u_current)

        # Update Values

        u_old = u_current
        u_current = u_true
        counter +=1

    return u  

u = continuation(PPM,[0.57,0.31,21],[1,0.2,0.1])

# %%

plt.plot(u[0,:],u[1,:])
# %%
