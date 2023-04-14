#%%

import numpy as  np
from week19 import construct_A_and_b, BoundaryCondition, Grid
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

def explicit_diffusion_solver(grid,bc_left,bc_right,D,t_steps):

    dx = grid.dx
    [A,b] = construct_A_and_b(grid,bc_left,bc_right)
    u = np.zeros((len(grid.x)-2,t_steps+1))

    u_old = u[:,0]

    for n in range(t_steps):


        dt = 0.5*dx**2/D # Recalculate dt to ensure stability
        t = dt*t_steps # Current time

        u_new = u_old + dt*D/dx**2*(np.matmul(A,u_old)+b)
        u_old = u_new

        u[:,n+1] = u_new
        
    return u


#%%

bc_left = BoundaryCondition('dirichlet', [0])
bc_right = BoundaryCondition('dirichlet', [6])
grid = Grid(N=100, a=0, b=10)
x = grid.x

u = explicit_diffusion_solver(grid,bc_left,bc_right,D=1,t_steps=1000)

plt.plot(grid.x[1:-1],u[:,-1], 'o', markersize = 2)

# %%

fig,ax = plt.subplots()

line, = ax.plot(x[1:-1],u[:,0])

def animate(i):
    line.set_data((x[1:-1],u[:,i]))
    return line,

ani = FuncAnimation(fig, animate, frames=1000, interval=20, blit=True)
plt.show()
# %%
