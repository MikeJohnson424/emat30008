
#%%

from scipy.optimize import fsolve, root
from integrate import solve_to
import numpy as np
import matplotlib.pyplot as plt
from functions import PPM
from PDEs import construct_A_and_b, Grid, BoundaryCondition


def shooting(func,init,parameters):

    x_0 = init[:-1]
    T = init[-1]
    sol = solve_to(func,x_0,T,parameters) 
    x_T = sol.x[:,-1]
    dxdt_0 = func(x_0,0,parameters)[0]
    
    return np.hstack((x_0 - x_T, dxdt_0))



def solve_BVP(func,bc_left,bc_right,parameters,T,method = 'shooting',grid= None):


    if not(bc_left.value == 'dirichlet' and bc_right.value == 'dirichlet'):
        raise ValueError('Both boundary conditions must be dirichlet')
    x0 = bc_left.value[0]
    x_final = bc_right.value[0]
    

    if grid.left != 0:
        raise ValueError('grid.right must be 0')

    if method == 'shooting':

        def shooting(func,IC,x_final,parameters):
            return solve_to(func,IC,T,parameters).x[:,-1] - x_final
        
        sol = root(lambda x: shooting(func,x,x_final,parameters),x0)
    
    elif method == 'finite-difference':

        [A, b] = construct_A_and_b(grid,bc_left,bc_right)

        u = np.linalg.solve(A,-b(t)-dx**2/D*q(grid.x[1:-1]))

    else:
        raise ValueError('method must be either shooting or finite-difference')




def find_lim_cycle_conditions(func,init,parameters):

    lim_cycle_initial_conditions = root(lambda x: shooting(func,x,parameters),init).x

    return lim_cycle_initial_conditions

# %%

lim_cycle_conditions = find_lim_cycle_conditions(PPM,[0.5,0.5,20],[1,0.5,0.1])
lim_cycle = solve_to(PPM,lim_cycle_conditions[:-1],lim_cycle_conditions[-1],[1,0.1,0.1])
plt.plot(lim_cycle.x[0],lim_cycle.x[1])

# %%
