
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable,Union,Optional,Tuple

def euler_step(deltat_max:float,x:np.ndarray, func:Callable,parameters:Optional[np.ndarray]=None,t:Union[float,None]=None) -> np.ndarray: # define function to compute a single euler step

    """
    A function that uses the forward Euler method to integrate over a single time-step.

    Parameters
    ----------
    deltat_max : float
        The time-step being iterated over.
    x : np.ndarray
        Conditions of ODE system at the current time level.
    func : Callable
        ODE system being integrated.
    parameters : np.ndarray
        Array containing the parameters of the ODE.
    t : Union[float, None], optional
        Current time level.

    Returns
    -------
    np.ndarray
        A NumPy array defining the solution at the next time-level.
    """

    # Solve for x at t+deltat_max:

    x_n1 = x + deltat_max*func(x,t,parameters)
    return x_n1

def RK4_step(deltat_max:float,x:np.ndarray,func:Callable,parameters:np.ndarray,t:Union[float,None]=None) -> np.ndarray:

    """
    A function that uses the Runge-Kutta 4 integration method over a single time-step.

    Parameters
    ----------
    deltat_max : float
        Size of time-step.
    x : np.ndarray
        Current conditions of the system.
    func : Callable
        ODE system to integrate.
    parameters : np.ndarray
        Array containing the parameters of the ODE.
    t : Union[float, None], optional
        Current time level.

    Returns
    -------
    np.ndarray
        A NumPy array that defines the solution at the next time-step.
    """

    # Define incremental function values:

    k1 = func(x,t,parameters)
    k2 = func(x+deltat_max*k1/2, t+deltat_max/2, parameters)
    k3 = func(x + deltat_max*k2/2, t+ deltat_max/2, parameters)
    k4 = func(x+deltat_max*k3, t+deltat_max, parameters)

    # Solve for next time-step:

    x_n1 = x + (deltat_max/6)*(k1+2*k2+2*k3+k4)

    return x_n1

class solve_to_result:
    """
    A class representing the result of an ODE integration using the `solve_to` function.

    Attributes
    ----------
    x : np.ndarray
        A numpy array containing the solution to the ODE system at each time-step.
    t_space : np.ndarray
        A numpy array containing the intermediate time-steps at which the ODE system was solved.
    """
    def __init__(self,x,t_space):
        self.x = x
        self.t_space = t_space

def solve_to(func:Callable, x0:np.ndarray, t:Tuple[float, float], parameters:np.ndarray=[], deltat_max:float=0.01, method:str = 'RK4') -> solve_to_result:

    """
    A function that iterates over a time-range using a chosen integration method to solve for the solution of a 
    given ODE.

    Parameters
    ----------
    func : Callable
        The ODE to solve to time t from time = 0. Must be of the form func(x, t, parameters) where x, parameters are NumPy arrays.
    x0 : np.ndarray
        Initial conditions for ODE system.
    t : Tuple[float, float]
        A tuple containing two values, t0 and t1, defining the time range to solve over.
    parameters : np.ndarray, optional
        Array containing the parameters of the ODE.
    deltat_max : float, optional
        Defines the maximum time step to be used in the integration.
    method : str, optional
        Choose from 'euler' or 'RK4' to define which solver is used

    Returns
    -------
    solve_to_result.t_space : np.ndarray
        A list of intermediate time steps for which ODE is solved at
    solve_to_result.x np.ndarray
        A numpy array containing solution across time to ODE system given x0.
    """

    t0,t1 = t # Unpack t

    # Define stepper function to be used in integration, return ValueError if method not recognised:

    if method == 'forward_euler':
        stepper = euler_step
    elif method =='RK4':
        stepper = RK4_step
    else:
        raise ValueError("Method must be either 'forward_euler' or 'RK4'")

    # Check for invalid time inputs

    if t0 < 0 or t1 < 0:
        raise ValueError("Time cannot be negative")
    elif t0 >= t1:
        raise ValueError("t1 must be greater than t0")
    else: pass

    # Check for invalid deltat_max input:

    if deltat_max > t1-t0:
        raise ValueError("deltat_max must be less than t1-t0")

    # Define arrays to store solution and iterate over time:

    x_old = np.array(x0) 
    t_space = np.arange(t0,t1+deltat_max,deltat_max) 
    t_space[-1] = t1 # Final time must be equal to user input t

    x = np.zeros([len(x_old),len(t_space)]) # Define array to store solution

    # Iterate over t_space using chosen stepper function:
      
    for counter, i in enumerate(t_space[:-1]):

        x[:,counter] = x_old
        x_new = stepper(deltat_max, x_old,func,parameters,i)
        x_old = x_new
        counter += 1

    # Complete final iteration where time-step != deltat_max:

    delta_t = t1 - t_space[-2]
    x[:,counter] = stepper(delta_t, x[:,-2],func,parameters,t1)


    return solve_to_result(x, t_space)
