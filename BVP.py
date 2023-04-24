
#%%

from scipy.optimize import fsolve, root
from integrate import solve_to
import numpy as np
import matplotlib.pyplot as plt
from functions import PPM
from PDEs import construct_A_and_b, Grid, BoundaryCondition
import scipy
import types


#%%

def lim_cycle_conditions(func,init,parameters):

    x_0 = init[:-1]
    T = init[-1]
    sol = solve_to(func,x_0,[0,T],parameters) 
    x_T = sol.x[:,-1]
    dxdt_0 = func(x_0,0,parameters)[0]
    
    return np.hstack((x_0 - x_T, dxdt_0))

def shooting(func,init,parameters,solver):
      
    """
    A function to solve for the required initial conditions and period of a limit cycle for a given ODE.
    Parameters
    ----------
    func: function
        The ODE to solve limit cycles for
    init: python list
        Array containing the initial guess for initial solution and period of limit cycle.
    parameters: python list
        Array containing the parameters of the ODE.
    solver: function
        The solver to use. Code currently only supports scipy.optimize.root.
    Returns
    -------
    Returns a an object with attributes:
    x: array
        Initial condition of limit cycle
    T: float
        Period of limit cycle
    """

    sol = solver(lambda x: lim_cycle_conditions(func,x,parameters),init)

    class result():
         
        def __init__(self,x):
            self.x = x[:-1]
            self.T = x[-1]

    return result(sol)

def BVP_solver(grid,bc_left,bc_right,IC,q,D,u_guess = None):
    """
    A function to solve boundary value problems for the time-invariant diffusion equation.
    Parameters
    -------
    grid: object
        Object returned by Grid function. Contains dx and x attributes.
    bc_left: object
        Object returned by BoundaryCondition function. Contains boundary condition type, value and A matrix entires.
    bc_right: object
        Object returned by BoundaryCondition function. Contains boundary condition type, value and A matrix entires.
    IC: Float, int or function
        Defines domain, x, at time t = 0.
    q: float, int or function q = q(x,u)
        Source term in reaction-diffusion equation.
    D: float or int
        Diffusion coefficient
    u_guess: float, int or function of x
        Initial guess for solution to if a non-linear source term is present
    Returns
    -------
    Returns a an object with attributes:
    x: array
        Initial condition of limit cycle
    T: float
        Period of limit cycle
    """
    dx = grid.dx
    x = grid.x
    t = None # T dependance not relevant in BVPs for time invariant diffusion equation

    # Adjust size of grid and u_guess if boundary conditions are dirichlet

    if bc_left.type == 'dirichlet':
        x = x[1:]
    if bc_right.type == 'dirichlet':
        x = x[:-1]

    # Initialise u_guess if not provided by user

    if isinstance(u_guess,type(None)):
        u_guess = np.zeros(len(x))
    elif type(u_guess) in (int,float):
        u_guess = u_guess*np.ones(len(x))
    elif isinstance(u_guess,types.FunctionType):
        u_guess = u_guess(x)
    else:
        raise TypeError('u_guess must be a float, int or function')

    # If q entered as float or int, convert to function
        
    if not isinstance(q,types.FunctionType):
        source = q
        q = lambda x,u: source

    A,b = construct_A_and_b(grid,bc_left,bc_right)

    # Use either finite difference or scipy.optimize.root to depending on if source term is linear or nonlinear

    if q(1,1) != q(1,2):
        sol = scipy.optimize.root(lambda u: D/dx**2*(A@u+b(t))+q(x,u),u_guess)
        if not sol.success:
            raise RuntimeError('Solver failed to converge, please choose a better guess for u')
        else:
            u = sol.x

    else:
        u = np.linalg.solve(A,-b(None)-dx**2/D*q(x,None))

    

    class result():

        def __init__(self,u,x):
                
                self.u = u
                self.x = x

    return result(u,x)

# %%

grid = Grid(5,0,1)
bc_left = BoundaryCondition('neumann',[2],grid)
bc_right = BoundaryCondition('dirichlet',[5],grid)
IC = lambda x: 0
q = lambda x,u: 0
D = 1

result = BVP_solver(grid,bc_left,bc_right,IC,q,D,5)
u = result.u
x = result.x
plt.plot(x,u)



# %%
