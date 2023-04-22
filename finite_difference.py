#%%

import numpy as  np
from math import floor
import matplotlib.pyplot as plt
import types
from integrate import solve_to
import scipy.sparse as sp
from matplotlib.animation import FuncAnimation 
import plotly.graph_objects as go
import sympy

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

    """
    A function that builds the A and b matrix depending on set boundary conditions

    Parameters
    ----------
    grid : Obj
        Contains discretized array,x and grid spacing dx.
    bc_left: Obj
        Object returned by BoundaryCondition function. Contains boundary condition type, value and A matrix entires.
    bc_right: Obj
        Object returned by BoundaryCondition function. Contains boundary condition type, value and A matrix entires.
    storage_type: Str
        Choose from dense or sparse to determine the nature of the A matrix.
    Returns
    -------
    Returns matrix A and function b_func. b_func is a function of time and defines the matrix b for a given time input.
    """

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

def du_dt(u, t, parameters):  # Define explicit temporal derivative of u
    A, b, q, D, dx, x = parameters
    if isinstance(A, np.ndarray):  # Check if A is a dense matrix
        return D / dx**2 * (np.dot(A, u) + b(t)) + q(x, t, u)
    elif sp.issparse(A):  # Check if A is a sparse matrix
        return D / dx**2 * (A.dot(u) + b(t)) + q(x, t, u)
    else:
        raise ValueError('Matrix A must be either dense (numpy.ndarray) or sparse (scipy.sparse)')

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
    IC: Float, int or function
        Defines domain, x, at time t = 0.
    D: float or int
        Diffusion coefficient
    q: float, int or function q = q(x,t,u)
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

    # Account for different types of source term
    
    if type(q) in (float,int):
        q = (lambda value: lambda x, t, u: value)(q)
    elif isinstance(q,types.FunctionType):
        pass
    else:
        raise TypeError('q must be a float, integer or some function q(x,t,u)')
    
    # Check if method conflicts with source term

    if q(1,2,3) != q(1,2,4) and method in ('implicit-euler','crank-nicolson'):
        raise Exception('Non-linear source terms cannot be used with chosen method.')

    # Adjust functions for solving matrix equations, generating identity matrix and matrix multiplication depending on storage type.

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

    if method in ('explicit-euler','lines'):
        dt = 0.5*dx**2/D # Recalculate dt to ensure stability if a time-step restriction is present

    # Set initial condition

    if isinstance(IC,types.FunctionType):
        u[:,0] = IC(x)
    elif type(IC) in (float,int):
        u[:,0] = IC*np.ones(len(grid.x))
    else:
        raise TypeError('Initial condition must be some function f(x) or constant')

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
            U = U + dt*du_dt(U, t, [A, b, q, D, dx, x])
            u[:,n+1] = U # Store solution

    elif method == 'lines':

        u = solve_to(du_dt, U, t = t_final,parameters=[A,b,q,D,dx,x],deltat_max = dt).x

    elif method == 'implicit-euler':

        for n in range(t_steps):

            t = n*dt+dt # Time at next time-level
            U = non_linear_solver( gen_eye(len(U))-C*A ,
                                    U+C*b(t)+dt*q(x,t,None)) # Solve for u_n+1 using implicit method
            u[:,n+1] = U # Store solution

    elif method == 'crank-nicolson':

        for n in range(t_steps):

            t_current = n*dt # Current time
            t_next = t_current + dt            
            U = non_linear_solver(gen_eye(len(U))-C/2*A,
                                  mat_mul((gen_eye(len(U))+C/2*A),U) + C/2*(b(t_current)+b(t_next)) + dt/2*(q(x,t_current,None)+q(x,t_next,None)))
            u[:,n+1] = U # Store solution

    elif method == 'IMEX':
        pass

    else:
        raise ValueError('Method not recognised')
    
    return u
    

#%%

""" GENERATE SOLUTION """

t_steps = 1000
storage = 'sparse'
D=1
dt = 0.1
method = 'crank-nicolson'

grid = Grid(N=100, a=0, b=1)
bc_left = BoundaryCondition('dirichlet', [lambda t: 0],grid)
bc_right = BoundaryCondition('dirichlet', [lambda t: 0],grid)
x = grid.x
q = 1
IC = 0


u = diffusion_solver(grid,
                    bc_left,
                    bc_right,
                    IC = 0,
                    D=0.1,
                    q=lambda x,t,u: 1,
                    dt=0.1,
                    t_steps=t_steps,
                    method='crank-nicolson',
                    storage = 'sparse')


#%%

""" ANIMATING SOLUTION """

if bc_left.type == 'dirichlet':
    x = x[1:]
if bc_right.type == 'dirichlet':
    x = x[:-1] 

fig,ax = plt.subplots()

line, = ax.plot(x,u[:,0])
ax.set_ylim(0,1)
ax.set_xlim(grid.left,grid.right)

def animate(i):
    line.set_data((x,u[:,i]))
    return line,

ani = FuncAnimation(fig, animate, frames=t_steps, interval=1, blit=True)
plt.show()


# %%

""" PLOT SOLUTION AS 3D SURFACE """

t_values = np.arange(0, (t_steps + 1) * dt, dt)
fig = go.Figure(data=[go.Surface(z=u, x=t_values, y=x)])

fig.update_layout(
    title='u(x,t)',
    autosize=False,
    scene=dict(
        xaxis=dict(range=[0, 3]),
        xaxis_title='t',
        yaxis_title='x',
        zaxis_title='u(x, t)'),
    width=500,
    height=500,
    margin=dict(l=65, r=50, b=65, t=90)
)

fig.show()

