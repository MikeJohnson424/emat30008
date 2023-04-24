
#%%

from scipy.optimize import fsolve, root
from integrate import solve_to
import numpy as np
import matplotlib.pyplot as plt
from functions import PPM
from PDEs import construct_A_and_b, Grid, BoundaryCondition
from scipy.integrate import solve_ivp


def shooting(func,init,parameters):

    x_0 = init[:-1]
    T = init[-1]
    sol = solve_to(func,x_0,[0,T],parameters) 
    x_T = sol.x[:,-1]
    dxdt_0 = func(x_0,0,parameters)[0]
    
    return np.hstack((x_0 - x_T, dxdt_0))

#%%


def BVP(init,BC,func,parameters):

    x0 = init[:-1]
    T = init[-1]
    
    x_final = solve_ivp(lambda t,x: func(x,None,parameters),[0,T],x0).y[:,-1]

    R1 = x_final - x0

    if BC == x0:
        R2 = func(x0,0,parameters)[0]
    else:
        R2 = T-T

    return np.array([R1,R2])

def shooting(func,init,parameters,BC):

    root(lambda x: shooting(init, BC, func, parameters),init).x
    
    return 

solution = shooting(PPM,[0.5,0.3,25],[1,0.1,0.1],[0.5,0.3,25])





#%%

def find_lim_cycle_conditions(func,init,parameters):

    lim_cycle_initial_conditions = root(lambda x: shooting(func,x,parameters),init).x

    return lim_cycle_initial_conditions

# %%
