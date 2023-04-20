#%%

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

    k1 = func(x,t,parameters)
    k2 = func(x+deltat_max*k1/2, t+deltat_max/2, parameters)
    k3 = func(x + deltat_max*k2/2, t+ deltat_max/2, parameters)
    k4 = func(x+deltat_max*k3, t+deltat_max, parameters)
    x_n1 = x + (deltat_max/6)*(k1+2*k2+2*k3+k4)
    return x_n1

def solve_to(func, x0, t, parameters, deltat_max=0.01, method = 'RK4'):

    """
    A function that iterates over a time-range using a chosen integration method to solve for the solution of a 
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
    method : String
        Choose from 'forward_euler' or 'RK4' to define which solver is used

    Returns
    -------
    Returns a python class with attributes t_space : a list of intermediate time steps for which ODE is solved at, and x : 
    a numpy array containing solution across time to ODE system given x0.
    """

    if method == 'forward_euler':
        stepper = euler_step
    elif method =='RK4':
        stepper = RK4_step

    if t <= 0:

        raise ValueError("Time cannot be negative!") # Check for negative input in time and throw error

    counter = 0
    x_old = np.array(x0)
    t_space = np.arange(0,t+deltat_max,deltat_max)
    t_space[-1] = t # Ensure that t_space always includes limits of t defined by user

    x = np.zeros([len(x_old),len(t_space)])
      
    for i in t_space[0:-1]:

        x[:,counter] = x_old
        x_new = stepper(deltat_max, x_old,func,parameters,i)
        x_old = x_new
        counter += 1

    # Complete final iteration where time-step != deltat_max:

    delta_t = t - t_space[-2]
    x[:,counter] = stepper(delta_t, x[:,-2],func,parameters,t)

    # Define class to return result:

    class result:
        def __init__(self,x,t_space):
            self.x = x
            self.t_space = t_space

    return result(x, t_space)


#%%

def f(x,t,paramters=[]):

    return (x**2+t**2)

result = solve_to(func = f,x0 = [1],t=0.4,parameters = [], deltat_max = 0.01, method='forward_euler')

#plt.plot(result.t_space,result.x[0])
#print(result.x[0,-1])

#print('e ~ ' + str(result.x[0,-1]))

# %%
