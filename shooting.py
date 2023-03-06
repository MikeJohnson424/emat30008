#%%

from scipy.optimize import fsolve
from integrate import solve_to
import numpy as np


def lim_cycle_problem(func,init):

    x0 = init[0:-1]
    T = init[-1]
    sol = solve_to(func,x0,T).x[:,-1]
    phase_con = func(x0)[0]
    phase_con1 = x0[0] - 0.4

    return np.hstack((x0-sol,phase_con))


def isolate_lim_cycle(func, init):

    """
    A function to be used to isolate a limit cycle by finding suitable initial conditions
    and its period

    Parameters
    ----------
    func : function
        The ODE to apply shooting to. The ode function should take
        a single parameter (the state vector) and return the
        right-hand side of the ODE as a numpy.array.
    init : python list
        An initial guess at the initial values for the limit cycle. This input is a list defined as [x0, y0, T] where
        x0, y0 are initial conditions and T is period.

    Returns
    -------
    Returns a numpy.array containing the corrected initial values
    for the limit cycle. If the numerical root finder failed, the
    returned array is empty.
    """

    ans = fsolve(lambda x: lim_cycle_problem(func, x),init)

    x0 = ans[0:-1]
    period = ans[-1]
    
    class ans:
        def __init__(self, period, x0):
            self.period = period
            self.x0 = x0

    return ans(period,x0)

# %%


