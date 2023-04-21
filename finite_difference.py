#%%

import numpy as  np
from math import floor
import matplotlib.pyplot as plt
import types
from integrate import solve_to
import scipy.sparse as sp
from matplotlib.animation import FuncAnimation 
import plotly.graph_objects as go

def gen_diag_mat(N,entries):

    """
    A function that uses scipy.sparse.diags to generate a diagonal matrix

    Parameters
    ----------
    N : Int
        Size of matrix is NxN
    entries : Python list
        Entries to be placed on the diagonals of the matrix.
    storage_type : String
        Determines the storage type of the matrix. Can be 'dense' or 'sparse'.

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
    mat = sp.diags(k,diagonals).toarray()

    return mat

def Grid(N=10,a=0,b=1):

    """
    A function that generates various attributes used in diffusion_solver

    Parameters
    ----------
    N : Int
        Number of discrete points in domain, x, is N+1
    a : Float or int
        Lower limit of domain x
    b : Flat or int
        Upper limit of domain x

    Returns
    -------
    grid.x : Numpy array
        Array of grid points
    grid.dx : Float
        Grid spacing
    grid.left : Float
        Lower limit of domain x
    grid.right : Float
        Upper limit of domain x
    """
    x = np.linspace(a,b,N+1)
    dx = (b-a)/N

    class grid:
        def __init__(self,x,dx,left,right):
            self.x = x
            self.dx = dx
            self.left = left
            self.right = right

    return grid(x,dx,a,b)

def BoundaryCondition(bcon_type, value,grid):
    
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
    grid: object
        An object containing the grid spacing and limits of the domain.
    Returns
    -------
    Returns a class containing the modified A entry, value and type of boundary condition.
    """

    dx = grid.dx

    if bcon_type == 'dirichlet':
        if type(value) != list or len(value) != 1 or isinstance(value[0], types.FunctionType) == False:
            raise ValueError('Dirichlet condition requires a python list containing a lambda function.')
        A_entry = [-2,1]

    elif bcon_type == 'neumann':
        if type(value) != list or len(value) != 1 or isinstance(value[0], types.FunctionType) == False:
            raise ValueError('Neumann condition requires a python list containing a lambda function')
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

def construct_A_and_b(grid,bc_left,bc_right,storage_type = 'dense'):

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

    if storage_type == 'sparse': # Convert A matrix to sparse matrix
        A = sp.csr_matrix(A)

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

    return 0

def du_dt(u, t, parameters):  # Define explicit temporal derivative of u
    A, b, q, D, dx, x = parameters
    if isinstance(A, np.ndarray):  # Check if A is a dense matrix
        return D / dx**2 * (np.dot(A, u) + b(t)) + q(x, t, u)
    elif sp.issparse(A):  # Check if A is a sparse matrix
        return D / dx**2 * (A.dot(u) + b(t)) + q(x, t, u)
    else:
        raise ValueError('Matrix A must be either dense (numpy.ndarray) or sparse (scipy.sparse)')

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
                    D,
                    q,
                    dt=10,
                    t_steps=20,
                    method='implicit-euler',
                    storage = 'dense'):
    
    """
    A function that iterates over a time-range using a chosen finite difference method
    to solve for the solution u(x,t) to a diffusive equation given dirichlet, neumann or robin boundary conditions.

    Parameters
    ----------
    grid: object
        Object returned by Grid function. Contains dx and x attributes.
    bc_left: object
        Object returned by BoundaryCondition function. Contains boundary condition type, value and A matrix entires.
    bc_right: object
        Object returned by BoundaryCondition function. Contains boundary condition type, value and A matrix entires.
    IC: object
        Object returned by InitialCondition function. Contains a function or float.
    D: float or int
        Diffusion coefficient
    q: function(x,t,u)
        Source term in reaction-diffusion equation.
    dt : float
        Used in implicit method to determine time step size. (Input for dt not considered in explicit method as dt is
        recalculated for stability).
    t_steps : int
        Number of time steps to iterate over.
    method : string
        Finite difference method. Choose from "lines", "explicit-euler", "implicit-euler", "crank-nicolson", "IMEX".
    storage : string
        Choose 'dense' or 'sparse' to set how A matrix is stored in memory. Default is 'dense'.
    
    Returns
    -------
    Returns a numpy array containing the solution at each time step.
    """
    
    dx = grid.dx # Grid spacing
    x = grid.x # Array of grid points
    C = dt*D/dx**2 
    [A,b] = construct_A_and_b(grid,bc_left,bc_right,storage) # Construct A and b matrices
    u = np.zeros((len(grid.x),t_steps+1)) # Initialise array to store solutions
    t_final = dt*t_steps # Final time

    # Adjust functions for solving matrix equations, generating identity matrix and
    # matrix multiplication depending on storage type.

    if storage == 'dense':
        non_linear_solver = np.linalg.solve
        gen_eye = np.eye
        mat_mul = np.matmul
    elif storage == 'sparse':
        non_linear_solver = sp.linalg.spsolve
        gen_eye = sp.eye
        mat_mul = lambda A,b: A.dot(b)
    else:
        raise ValueError('Storage type must be "dense" or "sparse"')

    if method == 'explicit-euler' or 'lines':
        dt = 0.5*dx**2/D # Recalculate dt to ensure stability if a time-step restriction is present

    # Set initial condition

    if IC.IC_type == 'function':
         u[:,0] = IC.initial_condition(x) # Set initial condition
    if IC.IC_type == 'constant':
        u[:,0] = IC.initial_condition*np.ones(len(grid.x)) # Set initial condition

    # Remove rows from solution matrix if dirichlet boundary conditions are used

    if bc_left.type == 'dirichlet':
        u = u[:-1]; x = x[1:]
    if bc_right.type == 'dirichlet':
        u = u[1:]; x = x[:-1] 

    U = u[:,0] # Set old solution as initial condition

    # Solve for solution using chosen method

    if method == 'explicit-euler':

        for n in range(t_steps):

            dt = 0.5*dx**2/D # Recalculate dt to ensure stability
            t = dt*n # Current time
            U = U +dt*du_dt(U,t,[A,b,q,D,dx,x]) # Time march solution
            u[:,n+1] = U # Store solution

    elif method == 'lines':

        u = solve_to(du_dt, U, t = t_final,parameters=[A,b,q,D,dx,x],deltat_max = dt).x

    elif method == 'implicit-euler':

        for n in range(t_steps):

            t = n*dt # Current time
            U = non_linear_solver(gen_eye(len(U))-C*A,
                                    U+C*b(t)) # Solve for u_n+1 using implicit method
            u[:,n+1] = U # Store solution

    elif method == 'crank-nicolson':

        for n in range(t_steps):

            t = n*dt # Current time
            U = non_linear_solver(gen_eye(len(U))-C/2*A,
                                    mat_mul((gen_eye(len(U))+C/2*A),U)+np.dot(C,b(t)))
            u[:,n+1] = U # Store solution

    elif method == 'IMEX':
        pass

    else:
        raise ValueError('Method not recognised')
    
    return u
    

#%%

""" GENERATE SOLUTION """

grid = Grid(N=10, a=0, b=1)
bc_left = BoundaryCondition('dirichlet', [lambda t: 0],grid)
bc_right = BoundaryCondition('dirichlet', [lambda t: 0],grid)
IC = InitialCondition(lambda x: 10*np.sin(np.pi*x))


x = grid.x
t_steps = 1000
storage = 'sparse'
D=1
dt = 0.1
method = 'explicit-euler'


u = diffusion_solver(grid,
                    bc_left,
                    bc_right,
                    IC,
                    D=0.1,
                    q=q,
                    dt=0.1,
                    t_steps=t_steps,
                    method='crank-nicolson',
                    storage = 'sparse')


#%%

# """ ANIMATING SOLUTION """

# if bc_left.type == 'dirichlet':
#     x = x[1:]
# if bc_right.type == 'dirichlet':
#     x = x[:-1] 

# fig,ax = plt.subplots()

# line, = ax.plot(x,u[:,0])
# ax.set_ylim(0,10)
# ax.set_xlim(grid.left,grid.right)

# def animate(i):
#     line.set_data((x,u[:,i]))
#     return line,

# ani = FuncAnimation(fig, animate, frames=t_steps, interval=10, blit=True)
# plt.show()


# %%

""" PLOT SOLUTION AS 3D SURFACE """

fig = go.Figure(data=[go.Surface(z=u, x=x, y=np.arange(0,(t_steps+1)*dt,dt))])
fig.update_layout(title='u(x,t)', autosize=False,
                  scene=dict(
        xaxis_title='t',
        yaxis_title='x',
        zaxis_title='u(x, t)'),
                  width=1000, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
# %%
