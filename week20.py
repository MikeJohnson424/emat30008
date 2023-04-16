#%%

import numpy as  np
from week19 import construct_A_and_b, BoundaryCondition, Grid
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import types

def q(x,t,u=None):

    array_size = len(x)
    time_array = np.full((array_size,1), t)

    return np.add(x, time_array)

def InitialCondition(initial_condition): 

    """
    A function that defines attributes of a chosen initial condition

    Parameters
    ----------
    initial_condition : Can be a function or a constant. If function then use lambda anonymous functions.
    
    Returns
    -------
    Returns a class containing a string specifying the nature of the intiial condition, and the initial condition itself.
    """

    if isinstance(initial_condition, types.FunctionType):
        IC_type = 'function'

    elif type(initial_condition) == float or type(initial_condition) == int:
        IC_type = 'constant'

    else:
        raise ValueError('Initial condition must be a function or a constant')

    class IC:
        def __init__(self,IC_type,initial_condition):
            self.IC_type = IC_type
            self.initial_condition = initial_condition
            

    return IC(IC_type,initial_condition)

def explicit_diffusion_solver(grid,bc_left,bc_right,IC,D,t_steps):

    dx = grid.dx # Grid spacing
    x = grid.x
    [A,b] = construct_A_and_b(grid,bc_left,bc_right) # Construct A and b matrices
    u = np.zeros((len(grid.x)-2,t_steps+1)) # Initialise array to store solutions

    if IC.IC_type == 'function':
         
         u_old = IC.initial_condition(x[1:-1]) # Set initial condition

    if IC.IC_type == 'constant':
            
        u_old = IC.initial_condition*np.ones(len(grid.x)-2) # Set initial condition

    for n in range(t_steps):

        dt = 0.5*dx**2/D # Recalculate dt to ensure stability
        t = dt*t_steps # Current time

        u_new = u_old + dt*D/dx**2*(np.matmul(A,u_old)+b) #+ dt*q(x,t,u_old) # Time march solution
        u_old = u_new

        u[:,n+1] = u_new # Store solution
        
    return u


#%%


grid = Grid(N=100,a = 0,b = 10)
bc_left = BoundaryCondition('dirichlet',[5])
bc_right = BoundaryCondition('dirichlet',[10])
IC = InitialCondition(lambda x: np.exp(x))
x = grid.x
dx = grid.dx
t_steps = 10000

u = explicit_diffusion_solver(grid,bc_left,bc_right,IC,D=1,t_steps=t_steps)

plt.plot(grid.x[1:-1],u[:,-1], 'o', markersize = 2)

# %%
""" ANIMATING SOLUTION """

fig,ax = plt.subplots()

line, = ax.plot(x[1:-1],u[:,0])
ax.set_ylim(0,1000)

def animate(i):
    line.set_data((x[1:-1],u[:,i]))
    return line,

ani = FuncAnimation(fig, animate, frames=t_steps, interval=1, blit=True)
plt.show()
# %%
