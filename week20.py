#%%

import numpy as  np
from week19 import construct_A_and_b, BoundaryCondition, Grid
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import types
from integrate import solve_to

def q(x,t,u):

    return np.zeros(len(x))

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

def explicit_diffusion_solver(grid,
                              bc_left,
                              bc_right,
                              IC,
                              D,
                              t_steps,
                              method = 'euler'):

    dx = grid.dx # Grid spacing
    x = grid.x
    [A,b] = construct_A_and_b(grid,bc_left,bc_right) # Construct A and b matrices
    dt = 0.5*dx**2/D # Recalculate dt to ensure stability
    t_final = t_steps*dt # Final time
    u = np.zeros((len(grid.x),t_steps+1)) # Initialise array to store solutions

    def du_dt(u,t,parameters): # Define explicit temporal derivative of u
        return D/dx**2*(np.matmul(A,u)+b(t)) + q(x,t,u)
    
    # Set initial condition

    if IC.IC_type == 'function':
         u[:,0] = IC.initial_condition(x) # Set initial condition
    if IC.IC_type == 'constant':
        u[:,0] = IC.initial_condition*np.ones(len(grid.x)) # Set initial condition

    # Remove rows from solution matrix and domain, x, if dirichlet boundary conditions are used

    if bc_left.type == 'dirichlet':
        u = u[:-1] 
        x = x[1:]
    if bc_right.type == 'dirichlet':
        u = u[1:]  
        x = x[:-1] 

    u_old = u[:,0] # Set initial condition as old solution 

    # Solve PDE depending on method set by user

    if method == 'lines':
    
        u = solve_to(du_dt, u_old, t = t_final,parameters=[],deltat_max=dt).x

    elif method == 'euler':

        for n in range(t_steps):

            dt = 0.5*dx**2/D # Recalculate dt to ensure stability
            t = dt*n # Current time
            u_new = u_old +dt*du_dt(u_old,t,parameters=[]) # Time march solution
            u_old = u_new # Update old solution

            u[:,n+1] = u_new # Store solution
        
    return u


#%%


grid = Grid(N=10,a = 0,b = 1)
x = grid.x
bc_left = BoundaryCondition('dirichlet',[lambda t: 5])
bc_right = BoundaryCondition('dirichlet',[lambda t: 10])
IC = InitialCondition(lambda x: np.zeros(len(x)))
dx = grid.dx
t_steps = 10000
D=1

u = explicit_diffusion_solver(grid,bc_left,bc_right,IC,D=1,t_steps=t_steps,method='lines')

plt.plot(grid.x[1:-1],u[:,-1], 'o', markersize = 2)

# %%

""" ANIMATING SOLUTION """

fig,ax = plt.subplots()

line, = ax.plot(x[1:-1],u[:,0])
ax.set_ylim(0,10)
ax.set_xlim(0,1)

def animate(i):
    line.set_data((x[1:-1],u[:,i]))
    return line,

ani = FuncAnimation(fig, animate, frames=t_steps, interval=50, blit=True)
plt.show()
# %%
