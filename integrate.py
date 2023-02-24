#%%

import numpy as np
from ODEs import g


def euler_step(deltat_max,x, func,t=0): # define function to compute a single euler step

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

    x_n1 = x + deltat_max*func(x)
    return x_n1

def RK4_step(deltat_max,x,func,t=0):

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

    k1 = func(x,t)
    k2 = func(x+deltat_max*k1/2, t+deltat_max/2)
    k3 = func(x + deltat_max*k2/2, t+ deltat_max/2)
    k4 = func(x+deltat_max*k3, t+deltat_max)
    x_n1 = x + (deltat_max/6)*(k1+2*k2+2*k3+k4)
    return x_n1

def solve_to(func, x0, t, deltat_max=0.01, stepper = RK4_step):

    """
    A function that iterates over an time-range using a chosen integration method to solve for the solution of a 
    given ODE.

    Parameters
    ----------
    func : function
        The ODE to solve to time t from time = 0.
    x0 : Python List
        Initial conditions for ODE system
    t : Float
        Time to integrate up until from t = 0
    deltat_max : Float
        Defines the maximum time step to be used in the integration
    stepper : Function
        Choose from euler_step or RK4_step to define which solver is used

    Returns
    -------
    Returns a python class with attributes t_space : a list of intermediate time steps for which ODE is solved at, and x : 
    a numpy array containing solution across time to ODE system given x0.
    """

    counter = 0
    x_old = np.array(x0)
    t_space = np.arange(0,t+deltat_max,deltat_max)
    t_space[-1] = t

    x = np.zeros([len(x_old),len(t_space)])
      
    for i in t_space[0:-1]:
        x[:,counter] = x_old
        x_new = stepper(deltat_max, x_old,func, i)
        x_old = x_new
        counter += 1

    # final iteration where time-step =/= deltat_max:

    delta_t = t - t_space[-2]
    x[:,counter] = stepper(delta_t, x[:,-2],func,t)

    class result:
        def __init__(self,x,t_space):
            self.x = x
            self.t_space = t_space

    return result(x, t_space)


#%%

def f(x,t=0): # Define ODE: dxdt = x, solution: x(t) = x0*exp(x)
    return(x)

result = solve_to(func = f,x0 = [1],t = 10, deltat_max = 0.1, stepper = RK4_step)
# %%
