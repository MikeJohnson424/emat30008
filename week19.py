#%%

import numpy as np
from scipy.sparse import diags
from math import floor
import matplotlib.pyplot as plt


def gen_diag_mat(N,entries):

    """
    A function that uses scipy.sparse.diags to generate a diagonal matrix

    Parameters
    ----------
    N : Integer
        Size of matrix is NxN
    entries : Python list
        Entries to be placed on the diagonals of the matrix.

    Returns
    -------
    Returns a numpy matrix of size NxN with the entries placed on the diagonals.
    """

    length = len(entries) # Number of diagonals
    lim = floor(length/2) # +/- limits of diagonals
    diagonals = range(-lim,lim+1) # Which diagonals to put the entries in

    k = [[] for _ in range(length)] # Create a list of empty lists

    for i in range(length):
        k[i] = entries[i]*np.ones(N - abs(diagonals[i])) # Fill the lists with the entries
    mat = diags(k,diagonals).toarray() # Create the N-diagonal matrix

    return mat

def Grid(N = 10, a = 0, b = 1):

    x = np.linspace(a,b,N+1)
    dx = (b-a)/N

    class grid:
        def __init__(self,x,dx):
            self.x = x
            self.dx = dx

    return grid(x,dx)

def BoundaryCondition(bcon_type, value):
    
    """
    A function that defines various features of the A matrix in the finite difference matrix equation

    Parameters
    ----------
    bcon_type : String
        Determine the nature of the boundary condition
    Value : Python list
        The associated values corresponding to the boundary condition. 
        For example, for a Dirichlet condition, the value is the value 
        of the function at the boundary. For a Neumann condition, the 
        value is the derivative of the function at the boundary. For 
        a Robin condition, the value is a list containing the delta and gamma values.

    Returns
    -------
    Returns a class containing the modified A entry, value and type of boundary condition.
    """

    if bcon_type == 'dirichlet':
        bcon_type = 'dirichlet'
        A_entry = [-2,1]

    elif bcon_type == 'neumann':
        A_entry = [-2,2]

    elif bcon_type == 'robin':
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

    x = grid.x
    dx = grid.dx
    N = len(x)-1
    b = np.zeros(N+1)

    # Set entries at either end of b vector (if dirichlet condition set then entries are not considered in calculation)

    b[0] = 2*bc_left.value[0]*dx
    b[-1] = 2*bc_right.value[0]*dx 

    A = gen_diag_mat(N+1,[1,-2,1])

    # Change A entries depending on boundary conditions set

    A[0,:2] = bc_left.A_entry
    A_entry_right = bc_right.A_entry
    A_entry_reversed = A_entry_right[::-1]
    A[-1,-2:] = A_entry_reversed

    # Delete first column and row, update b vector if left boundary condition is dirichlet

    if bc_left.type == 'dirichlet':

        A = A[1:,1:]
        b = b[1:]
        b[0] = bc_left.value[0]

    # Delete last column and row, update b vector if right boundary condition is dirichlet

    if bc_right.type == 'dirichlet':

        A = A[:-1,:-1]
        b = b[:-1]
        b[-1] = bc_right.value[0]
    

    return [A, b]

def q(x):

    return np.ones(len(x))

def sol_no_source(x,a,b,alpha,beta):

    return ((beta-alpha))/(b-a)*(x-a)+alpha

def sol_source(x,a,b,alpha,beta,D):

    return -1/(2*D)*(x-a)*(x-b)+(beta-alpha)/(b-a)*(x-a)+alpha

# %%


bc_left = BoundaryCondition('dirichlet', [5])
bc_right = BoundaryCondition('dirichlet',[10])
grid = Grid(N=100, a=0, b=10)
alpha = bc_left.value
beta = bc_right.value
dx = grid.dx
D = 0.1

[A, b] = construct_A_and_b(grid,bc_left,bc_right)
u = np.linalg.solve(A,-b-dx**2/D*q(grid.x[1:-1]))

# Plot results against true solution

#u_true_no_source = sol_no_source(grid.x,0,10,alpha,beta)
#u_true_source = sol_source(grid.x,0,10,alpha,beta,D)

#plt.plot(grid.x,u_true_source)
plt.plot(grid.x[1:-1], u, 'o', markersize = 2)

# %%
