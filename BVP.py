#%%

from scipy.optimize import root
from scipy.integrate import solve_ivp
import numpy as np
from functions import PPM, hopf_normal_form
from PDEs import construct_A_and_b
import types
from typing import Callable, Tuple, Union,Optional


def lim_cycle_conditions(func: Callable,init:np.ndarray,parameters:np.ndarray) -> np.ndarray:

    x0 = init[:-1]
    T = init[-1]
    sol = solve_ivp(lambda t, x: func(x,t,parameters),[0,T],x0)
    x_T = sol.y[:,-1]
    dxdt_0 = func(x0,0,parameters)[0]
    
    return np.hstack((x0 - x_T, dxdt_0))

class shooting_result():
         
        def __init__(self,x0,T):
            self.x0 = x0
            self.T = T

def shooting(func: Callable,init:np.ndarray,parameters:np.ndarray,solver:Callable=root) -> shooting_result:
      
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
    if not sol.success:
        raise Exception("Root finder failed to converge, please adjust parameters or init.")
    x0 = sol.x[:-1]
    T = sol.x[-1]
    

    return shooting_result(x0,T)

class BVP_solver_result():

    def __init__(self,u,x):
            
            self.u = u
            self.x = x

def BVP_solver(grid:object,bc_left:object,bc_right:object,q:Callable,D:Union[float,int],u_guess:Optional[np.ndarray] = None) -> BVP_solver_result:

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

    if isinstance(u_guess,types.NoneType):
        u_guess = np.zeros(len(x))
    elif isinstance(u_guess,(int,float)):
        u_guess = u_guess*np.ones(len(x))
    elif isinstance(u_guess,types.FunctionType):
        u_guess = u_guess(x)
    else:
        raise TypeError(f'u_guess must be a float, int, or function, not {type(u_guess)}')


    # If q entered as float or int, convert to function
        
    if not isinstance(q,types.FunctionType):
        source = q
        q = lambda x,u: source
    elif isinstance(q,types.FunctionType):
        pass
    else:
        raise TypeError('q must be a float, int or function q(x,u)')

    A,b = construct_A_and_b(grid,bc_left,bc_right)

    # Use either finite difference or scipy.optimize.root to depending on if source term is linear or nonlinear

    if q(1,1) != q(1,2):
        sol = root(lambda u: D/dx**2*(A@u+b(t))+q(x,u),u_guess)
        if not sol.success:
            raise RuntimeError('Solver failed to converge, please choose a better guess for u')
        else:
            u = sol.x
    else:
        u = np.linalg.solve(A,-b(None)-dx**2/D*q(x,None))

    return BVP_solver_result(u,x)


# %%
