#%%

import numpy as  np
from week19 import construct_A_and_b, BoundaryCondition, Grid
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def explicit_diffusion_solver(grid,bc_left,bc_right,D,dt,t_steps):

    dx = grid.dx
    [A,b] = construct_A_and_b(grid,bc_left,bc_right)
    u = np.zeros(len(grid.x)-2)

    for n in range(t_steps):

        t = dt*t_steps # Current time-stamp

        u_new = u + dt*D/dx**2*(np.dot(A,u)+b)
        u = u_new
        
    return u


#%%

bc_left = BoundaryCondition('dirichlet', 2)
bc_right = BoundaryCondition('dirichlet', 5)
grid = Grid(N=10, a=0, b=10)

u = explicit_diffusion_solver(grid,bc_left,bc_right,D=1,dt=0.1,t_steps=1000)

#plt.plot(grid.x[1:-1],u, 'o', markersize = 2)

# %%
