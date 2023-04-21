#%%

import numpy as  np
from week19 import construct_A_and_b, BoundaryCondition, Grid 
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def implicit_diffusion_solver(grid,bc_left,bc_right,D,dt,t_steps,method = 'euler'):

    dx = grid.dx
    x = grid.x
    C = dt*D/dx**2 
    dx = grid.dx # Grid spacing
    [A,b] = construct_A_and_b(grid,bc_left,bc_right) # Construct A and b matrices
    u = np.sin(np.pi*x[1:-1]) # Intial condition
    

    if method == 'euler':

        for n in range(t_steps):

            t = n*dt # Current time

            u_new = np.linalg.solve(np.eye(len(A))-C*A,u+np.dot(C,b(t))) # Solve for u(n+1) using implicit method
            u = u_new # Update u vector

    elif method == ' crank-nicolson':

        for n in range(t_steps):

            t = n*dt # Current time
            u_new = np.linalg.solve(np.eye(len(A))-C/2*A,
                                    np.matmul((np.eye(len(A))+C/2*A),u)+np.dot(C,b(t)))
            u = u_new # Update u vector
       
    return u

# %%

bc_left = BoundaryCondition('dirichlet', [lambda x: 0])
bc_right = BoundaryCondition('dirichlet', [lambda x: 0])
grid = Grid(N=100, a=0, b=1)

u = implicit_diffusion_solver(grid,bc_left,bc_right,D=0.1,dt=20,t_steps=10,method='crank-nicolson')
plt.plot(grid.x[1:-1],u, 'o', markersize = 2)

# %%


def crank_nicolson(grid,bc_left,bc_right,D,dt,t_steps):

    dx = grid.dx
    C = dt*D/dx**2 
    u = np.zeros(len(grid.x)-2) # Intialise u vector to zeros
    dx = grid.dx # Grid spacing

    [A,b] = construct_A_and_b(grid,bc_left,bc_right) # Construct A and b matrices


    return u
