
#%%

from scipy.optimize import fsolve, root
from integrate import solve_to
import numpy as np
import matplotlib.pyplot as plt
from functions import PPM


def shooting(func,init,parameters):

    x_0 = init[:-1]
    T = init[-1]
    sol = solve_to(func,x_0,T,parameters) 
    x_T = sol.x[:,-1]
    dxdt_0 = func(x_0,0,parameters)[0]
    
    return np.hstack((x_0 - x_T, dxdt_0))

def find_lim_cycle_conditions(func,init,parameters):

    lim_cycle_initial_conditions = root(lambda x: shooting(func,x,parameters),init).x

    return lim_cycle_initial_conditions

# %%

lim_cycle_conditions = find_lim_cycle_conditions(PPM,[0.5,0.5,20],[1,0.5,0.1])
lim_cycle = solve_to(PPM,lim_cycle_conditions[:-1],lim_cycle_conditions[-1],[1,0.1,0.1])
plt.plot(lim_cycle.x[0],lim_cycle.x[1])

# %%
