#%%

import numpy as  np
from week19 import construct_A_and_b, BoundaryCondition, Grid 
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from week20 import InitialCondition

def q(x,t,u):
    
    array_size = len(x)
    t = np.full((array_size,1),t).flatten()

    return np.ones(array_size)

def implicit_diffusion_solver(grid,bc_left,bc_right,IC,D,dt,t_steps,method = 'euler'):

    dx = grid.dx # Grid spacing
    x = grid.x # Array of grid points
    C = dt*D/dx**2 
    [A,b] = construct_A_and_b(grid,bc_left,bc_right) # Construct A and b matrices
    u = np.zeros((len(grid.x),t_steps+1)) # Initialise array to store solutions

    # Set initial condition

    if IC.IC_type == 'function':
         u[:,0] = IC.initial_condition(x) # Set initial condition
    if IC.IC_type == 'constant':
        u[:,0] = IC.initial_condition*np.ones(len(grid.x)) # Set initial condition

    # Remove rows from solution matrix if dirichlet boundary conditions are used

    if bc_left.type == 'dirichlet':
        u = u[:-1] 
        x = x[1:]
    if bc_right.type == 'dirichlet':
        u = u[1:]  
        x = x[:-1] 

    u_old = u[:,0] # Set initial condition as old solution 
    
    if method == 'euler':

        for n in range(t_steps):

            t = n*dt # Current time

            u_new = np.linalg.solve(np.eye(len(A))-C*A,
                                    u_old+C*b(t)) # Solve for u_n+1 using implicit method
            u_old = u_new # Update u vector

    elif method == 'crank-nicolson':

        for n in range(t_steps):

            t = n*dt # Current time
            u_new = np.linalg.solve(np.eye(len(A))-C/2*A,
                                    np.matmul((np.eye(len(A))+C/2*A),u_old)+np.dot(C,b(t)))
            u_old = u_new # Update u vector
    
    elif method == 'IMEX':

        for n in range(t_steps):

            t = n*dt # Current time
            u_new = np.linalg.solve(np.eye(len(A))-C*A,
                                    u_old+C*b(t)+C*q(x,t,u_old)) 
            u_old = u_new # Update u vector

    return u_old

# %%

bc_left = BoundaryCondition('dirichlet', [lambda x: 0])
bc_right = BoundaryCondition('dirichlet', [lambda x: 0])
IC = InitialCondition(lambda x: np.sin(np.pi*x))
grid = Grid(N=100, a=0, b=1)

u = implicit_diffusion_solver(grid,
                              bc_left,
                              bc_right,
                              IC,
                              D=0.1,
                              dt=0.01,
                              t_steps=1000,
                              method='IMEX')
plt.plot(grid.x[1:-1],u, 'o', markersize = 2)
plt.ylim(0,1000)
# %%