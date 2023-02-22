import numpy as np


def euler_step(deltat_max,x_n, f): # define function to compute a single euler step

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

    x_n1 = x_n + deltat_max*f(x_n)
    return x_n1

def RK_step(h,x,t,func): # h: time_step, x = current solution, t = current time, f = ODE function

    """
    A function that uses the runge-kutta 4 integration method over a single time-step.

    Parameters
    ----------
    h : float
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
    k2 = func(x+h*k1/2, t+h/2)
    k3 = func(x + h*k2/2, t+ h/2)
    k4 = func(x+h*k3, t+h)
    x_n1 = x + (h/6)*(k1+2*k2+2*k3+k4)
    return x_n1

def solve_to(func, x0, t, deltat_max=0.01, method='RK4'): # Method == 1: EULER, method == 2: RUNGE-KUTTA

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
    method : String
        Choose from 'forward euler' or 'RK4' to define which solver is used

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

    if method == 'forward_euler':  # Perform integration of ODE function using EULER METHOD in range t

        for i in t_space[0:-1]:
            x[:,counter] = x_old
            x_new = euler_step(deltat_max,x_old,f)
            x_old = x_new
            counter += 1

        # final iteration where time-step =/= deltat_max:

        deltat_final = t - t_space[-2]
        x[:,counter] = euler_step(deltat_final, x[:,-2], f)   

    else: # Perform integration of ODE function using RUNGE-KUTTA method in range t
            
        for i in t_space[0:-1]:
            x[:,counter] = x_old
            x_new = RK_step(deltat_max, x_old, i,func)
            x_old = x_new
            counter += 1

        # final iteration where time-step =/= deltat_max:

        delta_t = t - t_space[-2]
        x[:,counter] = RK_step(delta_t, x[:,-2], t,func)

    class result:
        def __init__(self,x,t_space):
            self.x = x
            self.t_space = t_space

    return result(x, t_space)
