
from scipy.optimize import fsolve, root, newton_krylov, anderson, linearmixing
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

