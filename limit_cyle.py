from scipy.optimize import fsolve
from integrate import solve_to
import numpy as np


def lim_cycle_problem(func,init):
    
    x0 = init[0:-1]
    T = init[-1]
    sol = solve_to(func,x0,T).x[:,-1]
    phase_con = func(x0)[0]

    return np.hstack((x0-sol,phase_con))


def isolate_lim_cycle(func, init):

    ans = fsolve(lambda x: lim_cycle_problem(func, x),init)

    return ans