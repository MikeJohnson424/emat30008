#%%

from scipy.optimize import fsolve
from integrate import solve_to
import numpy as np
from ODEs import PPM


def lim_cycle_problem(func,init):
    
    x0 = init[0:-1]
    T = init[-1]
    sol = solve_to(func,x0,T).x[:,-1]
    phase_con = func(x0)[0]
    phase_con1 = x0[0] - 0.4

    return np.hstack((x0-sol,phase_con))


def isolate_lim_cycle(func, init):

    ans = fsolve(lambda x: lim_cycle_problem(func, x),init)

    x0 = ans[0:-1]
    period = ans[-1]
    
    class ans:
        def __init__(self, period, x0):
            self.period = period
            self.x0 = x0

    return ans(period,x0)

# %%


period = isolate_lim_cycle(PPM, [0.5 , 0.3 , 21]).period
x0 = isolate_lim_cycle(PPM, [0.5 , 0.3 , 21]).x0
# %%
