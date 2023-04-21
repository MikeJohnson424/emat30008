#%%

import numpy as  np
from week19 import construct_A_and_b, BoundaryCondition, Grid
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import types
from integrate import solve_to





def diffusion_solver(grid,
                    bc_left,
                    bc_right,
                    IC,
                    D=0.1,
                    dt=20,
                    t_steps=20,
                    method='IMEX'):


    if method == 'explicit-euler':
        pass
    elif method == 'implicit-euler':
        pass
    elif method == 'crank-nicolson':
        pass
    elif method == 'lines':
        pass
    elif method == 'IMEX':
        pass
    else:
        raise ValueError('Method not recognised')