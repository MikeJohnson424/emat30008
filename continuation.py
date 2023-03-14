#%% 

import numpy as np
import matplotlib.pyplot as plt
from integrate import solve_to
from shooting import isolate_lim_cycle
from scipy.optimize import fsolve
from functions import PPM

def h(x,alpha):
    x = x
    return x**3-x-alpha


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

[x, alpha] = NP_continuation(h)

plt.plot(alpha,x[0])
plt.show()


# %%

""" == WORKING PSUEDO ARCLENGTH CONTINUATION FOR h(x,alpha) == """

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

u_h = PA_continuation(h)

plt.plot(u_h[1,:],u_h[0,:])

# %%


def solve_for(func, u, u_pred, delta_u):
    
    x,alpha = u

    top = func(x,alpha)
    bottom = np.dot((u - u_pred),delta_u)

    return np.array([top,bottom])

def PA_continuation(func,init=[x1,x2,x3,...],init_param=[a,b,c,d,...],par_idx = 0, imax=100, initial_step_size = 0.1): 

    u = np.zeros((len(init)+1,imax+1)) # Create array where solutions and corresponding parameter value will be stored

    alpha0 = init_param[par_idx] # Initial value of alpha
    x0 = isolate_lim_cycle(func, init).x0  # Find an initial solution
    u_old = np.hstack((x0,alpha0)) # Concatenate the initial solution and parameter value
    u[:,0] = u_old # Store initial solution and parameter value

    alpha1 = alpha0 + initial_step_size # Update alpha
    x1 = isolate_lim_cycle(func, init).x0 # Find a secondary solution in order to produce a secant
    u_current = np.hstack((x1,alpha1)) # Concatenate the secondary solution and parameter value

    counter = 1 # Initialize counter corresponding to index of u

    for i in range (imax): # Produce and store the rest of the solutions using pseudo-arclength continuation

        u[:,counter] = u_current

        # Linear Predictor: 

        delta_u = u_current - u_old # Calculate secant
        u_future_pred = u_current + delta_u # Predict the next solution

        # Corrector:

        u_future_true = fsolve(lambda u: solve_for(func,u, u_future_pred, delta_u),u_current) # Solve for true next solution

        # Update Values

        u[:,counter] = u_future_true
        u_old = u_current
        u_current = u_future_true
        counter +=1