#%% 

import numpy as np
import matplotlib.pyplot as plt
from integrate import solve_to
import functions
from shooting import isolate_lim_cycle
from scipy.optimize import fsolve


# %%

result = solve_to(functions.third_order_hopf, [1,1,1],10)
result2 = solve_to(functions.hopf_normal_form,[1,1],10)

x = result.x
t = result.t_space

ax = plt.axes(projection = '3d')
ax.plot3D(x[0],x[1],x[2])



# %%


# %%
