#%%

import numpy as np
import matplotlib.pyplot as plt
from shooting import shooting
from scipy.optimize import fsolve,root
from functions import PPM, h

#%%

def gen_sol_mat(x0,x_dim,max_steps):
    return np.zeros((x_dim+1,max_steps+1))

def predictor(u_current, u_old):
    delta_u = u_current - u_old
    u_pred = u_current + delta_u
    return [u_pred,delta_u]

def corrector(myode, u,u_pred,delta_u,vary_par,par0):

    par0[vary_par] = u[-1]
    R1 = myode(u[:-1],None,par0)
    R2 = np.dot(u - u_pred, delta_u)
    return np.hstack((R1,R2))

def find_initial_solutions(solver,myode,x0,par0,vary_par,step_size):

    u_old = np.hstack((
        solver(lambda x: myode(x,None,par0),x0).x,
        par0[vary_par]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    ))
    par0[vary_par] += step_size
    u_current = np.hstack((
        solver(lambda x: myode(x,None,par0),u_old[:-1]).x,
        par0[vary_par]
    ))

    return [u_old,u_current]

def continuation(myode,  # the ODE to use
    x0,  # the initial state
    par0,  # the initial parameters
    vary_par=0,  # the index of the parameter to vary
    step_size=0.1,  # the size of the steps to take
    max_steps=50,  # the number of steps to take
    discretisation=shooting,  # the discretisation to use
    solver=root):  # the solver to use
    
    x_dim = len(x0) # Dimension of the solution
    u = gen_sol_mat(x0,x_dim,max_steps) # Initialise matrix to contain solution and varying parameter

    u_old,u_current = find_initial_solutions(solver,myode,x0,par0,vary_par,step_size)

    u[:,0] = u_old # Store first solution

    for n in range(max_steps):

        u[:,n+1] = u_current

        # Linear Predictor

        u_pred,delta_u = predictor(u_current,u_old)

        # Corrector

        u_true = solver(lambda u: 
                        corrector(myode,u,u_pred,delta_u,vary_par,par0),
                          u_pred).x
        
        u_old = u_current
        u_current = u_true
        

    return u



# %%

""" TEST FOR h(x) """

y = np.linspace(-1.5,1.5,100)
plt.plot(y-y**3,y)

u = continuation(h,x0 = [1],par0 = [-2],
                   vary_par = 0,
                   step_size = 0.1,
                   max_steps = 50)
plt.plot(u[-1],u[0])

#%%

""" TEST FOR PPM """

u = continuation(PPM,x0 = [0.5,0.3],par0 = [0.5,0.2,0.1],
                   vary_par = 0,
                   step_size = 0.1,
                   max_steps = 50)
plt.plot(u[-1],u[0])

# %%

myode = PPM; x0 = [5,6]; par0 = [0.5,0.2,0.1]; vary_par = 0; step_size = 0.1; max_steps = 40; solver = root
#myode = h;x0 = [1];par0 = [-2];vary_par = 0;step_size = 0.1;max_steps = 50;solver = root

# %%
