
import numpy as np
import matplotlib.pyplot as plt
from BVP import lim_cycle_conditions, shooting
from scipy.optimize import root
from functions import PPM, h
from scipy.integrate import solve_ivp
from typing import Callable, List, Tuple, Union


def gen_sol_mat(x_dim: int,max_steps: int) -> np.ndarray:

    """
    A function for generating solution storage array as part of continuation.

    Parameters
    ----------
    x_dim: int
        
    max_steps: int
        
    Returns
    -------
    Returns a ndarray of zeros with dimensions (x_dim+1,max_steps+1) where x_dim is the dimension of the solution and max_steps is the number of steps to take.
    """
    return np.zeros((x_dim+1,max_steps+1))

def find_initial_solutions(solver:Callable,myode:Callable,x0:np.ndarray,par0:np.ndarray,vary_par:int,step_size:float,solve_for:str) -> Tuple[np.ndarray,np.ndarray]:

    """
    A function for generating two initial solutions to be used in pseudo-arclength continuation.

    Parameters
    ----------
    solver: function
        The solver to use. Code currently only support scipy.optimize.root.
    myode: function
        Function to perform continuation on.
    x0: ndarray
        The initial state.
    par0: ndarray
        The array of parameters.
    vary_par: int
        The index of the parameter being varied.
    step_size: float
        The size of the step to take.
    solve_for: str
        Choose from 'equilibria' or 'limit_cycle'. If 'equilibria' is chosen, the function will solve for equilibria. If 'limit_cycle' is chosen, the function will solve for limit cycle conditions.
    Returns
    -------
    u_old: ndarray
        First solution.
    u_current: ndarray
        Second solution.
    """

    # Checks for invalid inputs 

    if type(vary_par) != int:
        raise TypeError('par0 must be an int')  

    # Solve for chosen solution and concatenate with parameter value being varied
 
    if solve_for == 'limit_cycle': # Solve for limit cycle conditions using shooting method
        solution1 = shooting(myode,x0,par0,solver)
        u_old = np.hstack((
            #solver(lambda x: lim_cycle_conditions(myode,x,par0),x0).x,
            solution1.x0,
            solution1.T,
            par0[vary_par]
        ))   
        par0[vary_par] += step_size 
        solution2 = shooting(myode,u_old[:-1],par0,solver)
        u_current = np.hstack((
            #solver(lambda x: lim_cycle_conditions(myode,x,par0),u_old[:-1]).x,
            solution2.x0,
            solution2.T,
            par0[vary_par]
            )) 
        
    elif solve_for == 'equilibria': # Solve for equilibria using chosen root finding method
        u_old = np.hstack((
            solver(lambda x: myode(x,None,par0),x0).x,
            par0[vary_par]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        ))
        par0[vary_par] += step_size
        u_current = np.hstack((
            solver(lambda x: myode(x,None,par0),u_old[:-1]).x,
            par0[vary_par]
        ))
    
    else:
        raise ValueError('solve_for not recognised, please choose from equilibria or limit_cycle')
    
    return [u_old,u_current]

def predictor(u_current:np.ndarray,u_old:np.ndarray,method:str,step_size:float) -> Tuple[np.ndarray,np.ndarray]:

    """
    A function for performing the predictor step of pseudo-arclength continuation.

    Parameters
    ----------
    u_current: ndarray
        Array containing the current solution and the current value of the parameter being varied.
    u_old: ndarray
        Array containing the previous solution and the previous value of the parameter being varied.
    method: str
        Choose from 'nParam' for natural parameter or 'pArclength' for pseudo-arclength continuation
    step_size: float
        Size of steps taken when varying parameter.
    Returns
    -------
    u_pred: ndarray
        Array containing the predicted solution and the predicted value of the parameter being varied.
    delta_u: ndarray
        Array containing the difference between the current and previous solutions and the difference between the current and previous values of the parameter being varied.
    """

    if method == 'pArclength':
        delta_u = u_current - u_old
        u_pred = u_current + delta_u
    elif method == 'nParam':
        delta_u = np.hstack((np.zeros(len(u_old)-1),step_size))
        u_pred = u_current + delta_u 
    else:
        raise ValueError('Method not recognised, please choose from nParam or pArclength')     

    return [u_pred,delta_u]

def corrector(myode:Callable,u:np.ndarray,vary_par:int,par0:np.ndarray,solve_for:str,u_pred:np.ndarray,delta_u:np.ndarray) -> np.ndarray:

    """
    A function to solve as part of the corrector stage of pseudo-arclength continuation.

    Parameters
    ----------
    myode: function
        Function to perform continuation on.
    u: ndarray
        The variable to solve for as part of the corrector stage.
    vary_par: int
        The index of the parameter being varied.
    par0: ndarray
        The array of parameters.
    solve_for: str
        Choose from 'equilibria' or 'limit_cycle'. If 'equilibria' is chosen, the function will solve for equilibria. If 'limit_cycle' is chosen, the function will solve for limit cycle conditions.
    u_pred: ndarray
        The predicted solution and the predicted value of the parameter being varied.
    delta_u: ndarray
        The difference between the current and previous solutions and the difference between the current and previous values of the parameter being varied.
    Returns
    -------
    Returns a ndarray containing output of myode and the dot product between (u-u_pred) and delta_u given array u.
    """

    par0[vary_par] = u[-1] # Parameter is last entry in u

    if solve_for == 'limit_cycle':
        R1 = lim_cycle_conditions(myode,u[:-1],par0)
        R2 = np.dot(u - u_pred, delta_u)
    else:
        R1 = myode(u[:-1],None,par0)
        R2 = np.dot(u - u_pred, delta_u)

    return np.hstack((R1,R2))

class result():

        def __init__(self,u,alpha):
            
                self.u = u
                self.alpha = alpha

def continuation(myode:Callable,x0:np.ndarray,par0:np.ndarray,vary_par:int=0,step_size:float=0.1,max_steps:int=50,solve_for:str = 'equilibria', method:str='pArclength',solver:Callable=root) -> object:
    
    """
    A function for performing continuation.

    Parameters
    ----------
    myode: function
        Function to perform continuation on.
    x0: ndarray
        Guess for initial solution. If solve_for is 'equilibria', x0 should be a guess for the equilibrium. 
        If solve_for is 'limit_cycle', x0 should be a guess for the limit cycle conditions, i.e. [x0,T] 
        where x0 is a guess for a point on a limit cycle and T is the period of the limit cycle.
    par0: ndarray
        The array of parameters.
    vary_par: int
        The index of the parameter being varied.
    step_size: float
        The size of the steps to take. For pseudo-arclength this only applies to the initial step.
    max_steps: int
        The number of steps to take.
    solve_for: str
        Choose from 'equilibria' or 'limit_cycle'. If 'equilibria' is chosen, the function will solve for equilibria. If 'limit_cycle' is chosen, the function will solve for limit cycle conditions.
    method: str
        Choose from 'pArclength' or 'nParam'. If 'pArclength' is chosen, the function will use pseudo-arclength continuation. If 'nParam' is chosen, the function will use natural parameter continuation.
    solver: function
        The solver to use. Code currently only supports scipy.optimize.root.
        
    Returns
    -------
    result.u: ndarray
        Array of solution values
    result.alpha: ndarray
        Array of parameter values
    """

    x_dim = len(x0) # Dimension of the solution
    u = gen_sol_mat(x_dim,max_steps) # Initialise matrix to contain solution and varying parameter

    # Solve for and store initial two solutions

    u_old,u_current = find_initial_solutions(solver,myode,x0,par0,vary_par,step_size,solve_for)
    u[:,0] = u_old

    # Perform continuation

    for n in range(max_steps):

        u[:,n+1] = u_current # Store solution and parameter value

        # Predictor-Corrector

        u_pred,delta_u = predictor(u_current,u_old,method,step_size)
        u_true = solver(lambda u: 
                        corrector(myode,u,vary_par,par0,solve_for,u_pred,delta_u),
                        u_pred).x
        
        # Update values for next iteration

        u_old = u_current
        u_current = u_true


    return result(u[:-1,:],u[-1,:])
