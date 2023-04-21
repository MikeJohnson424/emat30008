#%%

import numpy as  np
from math import floor
import matplotlib.pyplot as plt
import types
from integrate import solve_to

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

def q(x,t,u):

    return np.zeros(len(x))

def du_dt(u,t,parameters): # Define explicit temporal derivative of u
        A,b,q,D,dx,x = parameters
        return D/dx**2*(np.matmul(A,u)+b(t)) + q(x,t,u)

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

def diffusion_solver(grid, 
                    bc_left, 
                    bc_right,
                    IC,
                    q,
                    D=0.1,
                    dt=None,
                    t_steps=20,
                    method='IMEX'):
    
    """
    A function that iterates over a time-range using a chosen finite difference method
    to solve for a solution u(x,t) to the diffusion equation.

    Parameters
    ----------
    grid: object
        
    bc_left: object
        
    bc_right: object
        
    IC: object

    q: function
        Source term.
    D: float or int
        Diffusion coefficient
    dt : float
        Used in implicit method to determine time step size. Dt is recalculated for stability in explicit methods
    t_steps : int
        Number of time steps to iterate over.
    method : string
    
    Returns
    -------
    Returns a numpy array containing the solution at each time step.
    """
    
    dx = grid.dx # Grid spacing
    x = grid.x # Array of grid points
    C = dt*D/dx**2 
    [A,b] = construct_A_and_b(grid,bc_left,bc_right) # Construct A and b matrices
    u = np.zeros((len(grid.x),t_steps+1)) # Initialise array to store solutions
    t_final = dt*t_steps # Final time

    if method == 'explicit-euler' or 'lines':
        dt = 0.5*dx**2/D # Recalculate dt to ensure stability if a time-step restriction is present

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

    u_old = u[:,0] # Set old solution as initial condition

    if method == 'explicit-euler':

        for n in range(t_steps):

            dt = 0.5*dx**2/D # Recalculate dt to ensure stability
            t = dt*n # Current time
            u_new = u_old +dt*du_dt(u_old,t,[A,b,q,D,dx,x]) # Time march solution
            u_old = u_new # Update old solution
            u[:,n+1] = u_new # Store solution

    elif method == 'lines':

        u = solve_to(du_dt, u_old, t = t_final,parameters=[]).x

    elif method == 'implicit-euler':

        for n in range(t_steps):

            t = n*dt # Current time
            u_new = np.linalg.solve(np.eye(len(A))-C*A,
                                    u_old+C*b(t)) # Solve for u_n+1 using implicit method
            u_old = u_new # Update u vector
            u[:,n+1] = u_new # Store solution

    elif method == 'crank-nicolson':

        for n in range(t_steps):

            t = n*dt # Current time
            u_new = np.linalg.solve(np.eye(len(A))-C/2*A,
                                    np.matmul((np.eye(len(A))+C/2*A),u_old)+np.dot(C,b(t)))
            u_old = u_new # Update u vector
            u[:,n+1] = u_new # Store solution

    elif method == 'IMEX':
        pass

    else:
        raise ValueError('Method not recognised')
    
    return u
    
#%%



