
import numpy as np
from functions import PPM
import matplotlib.pyplot as plt


def euler_step(deltat_max,x, func,parameters,t=None): # define function to compute a single euler step

    """
    A function that uses the forward euler method to integrate over a single time-step

    Parameters
    ----------
    deltat_max : Float
        The time-step being iterated over
    x_n : Python list
        Conditions of ODE system at current time level.
    f : function
        ODE system being integrated.

    Returns
    -------
    Returns a python list defining the solution at the next time-level.
    """

    # Solve for x at t+deltat_max:

    x_n1 = x + deltat_max*func(x,t)
    return x_n1

def RK4_step(deltat_max,x,func,parameters,t=None):

    """
    A function that uses the runge-kutta 4 integration method over a single time-step.

    Parameters
    ----------
    deltat_max : float
        Size of time-step
    x : Python list
        Current conditions of system.
    t : Float
        Current time level.
    func : function
        ODE system to integrate.

    Returns
    -------
    Returns a numpy array that defines the solution at the next time-step.
    """ 

    # Define incremental function values:

    k1 = func(x,t,parameters)
    k2 = func(x+deltat_max*k1/2, t+deltat_max/2, parameters)
    k3 = func(x + deltat_max*k2/2, t+ deltat_max/2, parameters)
    k4 = func(x+deltat_max*k3, t+deltat_max, parameters)

    # Solve for next time-step:

    x_n1 = x + (deltat_max/6)*(k1+2*k2+2*k3+k4)

    return x_n1

def solve_to(func, x0, t, parameters=[], deltat_max=0.01, method = 'RK4'):

    """
    A function that iterates over a time-range using a chosen integration method to solve for the solution of a 
    given ODE.

    Parameters
    ----------
    func : function
        The ODE to solve to time t from time = 0. Must be of the form func(x,t,parameters) where x,parameters are python lists.
    x0 : Python List
        Initial conditions for ODE system
    t : Python list
        List containing two values, t0 and t1, defining the time range to solve over.
    deltat_max : Float
        Defines the maximum time step to be used in the integration
    method : String
        Choose from 'forward_euler' or 'RK4' to define which solver is used

    Returns
    -------
    Returns a python class with attributes t_space : a list of intermediate time steps for which ODE is solved at, and x : 
    a numpy array containing solution across time to ODE system given x0.
    """

    t0,t1 = t # Unpack t

    # Define stepper function to be used in integration, return ValueError if method not recognised:

    if method == 'forward_euler':
        stepper = euler_step
    elif method =='RK4':
        stepper = RK4_step
    else:
        raise ValueError("Method must be either 'forward_euler' or 'RK4'")

    # Check for negative input in time and throw error:

    if t0 < 0:
        raise ValueError("Time cannot be negative!") 

    # Define arrays to store solution and iterate over time:

    counter = 0
    x_old = np.array(x0) 
    t_space = np.arange(t0,t1+deltat_max,deltat_max) 
    t_space[-1] = t1 # Final time must be equal to user input t

    x = np.zeros([len(x_old),len(t_space)]) # Define array to store solution

    # Iterate over t_space using chosen stepper function:
      
    for i in t_space[0:-1]:

        x[:,counter] = x_old
        x_new = stepper(deltat_max, x_old,func,parameters,i)
        x_old = x_new
        counter += 1

    # Complete final iteration where time-step != deltat_max:

    delta_t = t1 - t_space[-2]
    x[:,counter] = stepper(delta_t, x[:,-2],func,parameters,t1)

    # Define class to return result as object with attributes x and t_space:

    class result:
        def __init__(self,x,t_space):
            self.x = x
            self.t_space = t_space

    return result(x, t_space)
