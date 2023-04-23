#%%

import numpy as np
from scipy.sparse import diags
from math import floor
import matplotlib.pyplot as plt
from functions import sol_no_source, sol_source
import types
from continuation import continuation
from scipy.optimize import root
from functions import bratu
from PDEs import gen_diag_mat

def Grid(N = 10, a = 0, b = 1):

    x = np.linspace(a,b,N+1)
    dx = (b-a)/N

    class grid:
        def __init__(self,x,dx,left,right):
            self.x = x
            self.dx = dx
            self.left = left
            self.right = right

    return grid(x,dx,a,b)

def BoundaryCondition(bcon_type, value):
    
    """
    A function that defines various features of the A matrix in the finite difference matrix equation

    Parameters
    ----------
    bcon_type : String
        Determine the nature of the boundary condition
    Value : Python list
        Python list containing a lambda function. The associated values corresponding to the boundary condition. 
        For example, for a Dirichlet condition, the value is the value 
        of the function at the boundary. For a Neumann condition, the 
        value is the derivative of the function at the boundary. For 
        a Robin condition, the value is a list containing the delta and gamma values.
    
    Returns
    -------
    Returns a class containing the modified A entry, value and type of boundary condition.
    """

    if bcon_type == 'dirichlet':
        if type(value) != list or len(value) != 1:
            raise ValueError('Dirichlet condition requires a function or list containing a single value or lambda function.')
        A_entry = [-2,1]

    elif bcon_type == 'neumann':
        if type(value) != list or len(value) != 1 or isinstance(value[0], types.FunctionType) == False:
            raise ValueError('Neumann condition requires a list containing a lambda function')
        A_entry = [-2,2]

    elif bcon_type == 'robin':
        if type(value) != list or len(value) != 2:
            raise ValueError('Robin condition requires a list containing two values')
        A_entry = [-2*(1+value[1]*dx), 2] # Value = [delta, gamma]

    else:
        raise ValueError('Boundary condition type not recognized')

    class BC:
        def __init__(self,type,value, A_entry):
            self.type = bcon_type
            self.value = value
            self.A_entry = A_entry

    return BC(bcon_type,value, A_entry)

def construct_A_and_b(grid,bc_left,bc_right):

    x = grid.x # Domain of problem
    dx = grid.dx # Grid spacing
    N = len(x)-1 
    b = np.zeros(N+1) # Initlialize b vector    
    A = gen_diag_mat(N+1,[1,-2,1]) # Initialize A matrix

    # Change A entries depending on if boundary conditions are robin or neumann

    A[0,:2] = bc_left.A_entry
    A_entry_right = bc_right.A_entry;A_entry_reversed = A_entry_right[::-1]
    A[-1,-2:] = A_entry_reversed

    # Update A matrix and b vector if either boundary condition is dirichlet

    if bc_left.type == 'dirichlet':
        A = A[1:,1:]
        b = b[1:]
    if bc_right.type == 'dirichlet':
        A = A[:-1,:-1]
        b = b[:-1]

    # Define function that returns b vector

    def b_func(t):

        if type(bc_right.value[0]) != types.FunctionType:
            bc_left.value[0] = lambda t: bc_left.value[0]
        if type(bc_right.value[0]) != types.FunctionType:
            bc_right.value[0] = lambda t: bc_right.value[0]
            
        b[0] = 2*bc_left.value[0](t)*dx
        b[-1] = 2*bc_right.value[0](t)*dx 
    
        if bc_left.type == 'dirichlet':
            b[0] = bc_left.value[0](t)
        
        if bc_right.type == 'dirichlet':
            b[-1] = bc_right.value[0](t)

        return b


    return [A, b_func]

def q(x):

    return np.ones(len(x))


# %%

t = None
alpha = 5;beta = 10
bc_left = BoundaryCondition('dirichlet', [lambda t: 5]);bc_right = BoundaryCondition('dirichlet',[lambda t: 10])
grid = Grid(N=100, a=0, b=10)
dx = grid.dx
D = 0.1

[A, b] = construct_A_and_b(grid,bc_left,bc_right)

u = np.linalg.solve(A,-b(t)-dx**2/D*q(grid.x[1:-1]))

""" Plot results against true solution """

#u_true_no_source = sol_no_source(grid.x,0,10,alpha,beta)
u_true_source = sol_source(grid.x,0,10,alpha,beta,D)

plt.plot(grid.x,u_true_source)
plt.plot(grid.x[1:-1], u, 'o', markersize = 2)

# %%

