
#%%

import numpy as np
import matplotlib.pyplot as plt
from week14 import solve_to
from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp, solve_bvp


#%%


def hopf_normal(x,t=0,beta=1,sigma=1):
    x_dot0 = beta*x[0]-x[1]+sigma*x[0]*(x[0]**2+x[1]**2)
    x_dot1 = x[0] + beta*x[1] + sigma*x[1]*(x[0]**2+x[1]**2)
    return np.hstack((x_dot0,x_dot1))


# %%
