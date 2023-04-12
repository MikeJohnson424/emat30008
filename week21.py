#%%

import numpy as  np
from week19 import construct_A_and_b, BoundaryCondition, Grid
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def implicit_diffusion_solver(grid,bc_left,bc_right,D,dt,t_steps):

    dx = grid.dx
    C = dt*D/dx**2 
    dx = grid.dx # Grid spacing
    [A,b] = construct_A_and_b(grid,bc_left,bc_right) # Construct A and b matrices
    u = np.zeros(len(A)) # Intialise u vector to zeros

    for n in range(t_steps):

        u_new = np.linalg.solve(np.eye(len(A))-C*A,u+np.dot(C,b)) # Solve for u(n+1) using implicit method
        u = u_new # Update u vector

    return u

# %%

bc_left = BoundaryCondition('neumann', 2)
bc_right = BoundaryCondition('dirichlet', 5)
grid = Grid(N=10, a=0, b=10)

u = implicit_diffusion_solver(grid,bc_left,bc_right,D=1,dt=10,t_steps=10)
plt.plot(grid.x[:-1],u, 'o', markersize = 2)

# %%


def crank_nicolson(grid,bc_left,bc_right,D,dt,t_steps):

    dx = grid.dx
    C = dt*D/dx**2 
    u = np.zeros(len(grid.x)-2) # Intialise u vector to zeros
    dx = grid.dx # Grid spacing

    [A,b] = construct_A_and_b(grid,bc_left,bc_right) # Construct A and b matrices

    if type(bc_left.value) == float:

        for n in range(t_steps):

            if type(bc_left.value) == float:

            u_new = np.linalg.solve(np.eye(len(A)-C/2*A),
                                    np.dot(np.eye(len(A)+C/2*A),u)+C*b)
            
            u = u_new # Update u vector
    elif type(bc_left.value) == function

        for n in range(t_steps):

            u_new = np.linalg.solve(np.eye(len(A))-C/2*A,
                                    np.dot(np.eye(len(A))+C/2*A,u)+C*(b+b_new)
            
            u = u_new

    return u
