#%%
import numpy as np
import matplotlib.pyplot as plt
from shooting import shooting
from scipy.optimize import fsolve,root
from functions import PPM, h

#%%

#myode = PPM; x0 = [5,6]; par0 = [0.5,0.2,0.1]; vary_par = 0; step_size = 0.1; max_steps = 40; solver = root
myode = h;x0 = [1];par0 = [-2];vary_par = 0;step_size = 0.1;max_steps = 50;solver = root

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
counter = 0

# %%

u[:,counter+1] = u_current

delta_u = u_current - u_old
u_pred = u_current + delta_u

def objective(u, u_pred, par_dim, vary_par, delta_u):
    ode_residuals = myode(u[:-par_dim], None, u[-par_dim:])  # Shape (1,) for 1D ODEs

    if len(u[:-par_dim]) == 1:  # Check if the ODE is one-dimensional
        constraint_residuals = np.zeros(par_dim)
        constraint_residuals[0] = (u[0] - u_pred[0]) * delta_u[0]
        constraint_residuals[vary_par] = (u[-par_dim+vary_par] - u_pred[-par_dim+vary_par]) * delta_u[1]
    else:
        constraint_residuals = np.zeros(par_dim)
        constraint_residuals[:2] = (u[:2] - u_pred[:2]) * delta_u[:2]
        constraint_residuals[vary_par] = (u[-par_dim+vary_par] - u_pred[-par_dim+vary_par]) * delta_u[2]

    return np.hstack((ode_residuals, constraint_residuals))  # Shape (2,) for 1D ODEs


u_true = solver(lambda u: objective(u, u_pred, par_dim, vary_par, delta_u), u_pred).x

counter += 1
u_old = u_current
u_current = u_true 

# %%

""" DEFINING THE CONTINUATION FUNCTION """

def objective(u, u_pred, par_dim, vary_par, delta_u):
    ode_residuals = myode(u[:-par_dim], None, u[-par_dim:])  # Shape (1,) for 1D ODEs

    if len(u[:-par_dim]) == 1:  # Check if the ODE is one-dimensional
        constraint_residuals = np.zeros(par_dim)
        constraint_residuals[0] = (u[0] - u_pred[0]) * delta_u[0] - delta_u[0] * step_size
        constraint_residuals[vary_par] = (u[-par_dim+vary_par] - u_pred[-par_dim+vary_par]) * delta_u[1] - delta_u[1] * step_size
    else:
        constraint_residuals = np.zeros(par_dim)
        constraint_residuals[:2] = (u[:2] - u_pred[:2]) * delta_u[:2] - delta_u[:2] * step_size
        constraint_residuals[vary_par] = (u[-par_dim+vary_par] - u_pred[-par_dim+vary_par]) * delta_u[2] - delta_u[2] * step_size

    return np.hstack((ode_residuals, constraint_residuals))  # Shape (2,) for 1D ODEs

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

        u_true = solver(lambda u: objective(u, u_pred, par_dim, vary_par, delta_u), u_pred).x

        # Update values

        u_old = u_current
        u_current = u_true      

    return u

#%%

y = np.linspace(-1.5,1.5,100)
plt.plot(y-y**3,y)

u = continuation(h,x0 = [1],par0 = [-2],
                   vary_par = 0,
                   step_size = 0.2,
                   max_steps = 1000)
plt.plot(u[-1],u[0],alpha=0.5)
# %%
