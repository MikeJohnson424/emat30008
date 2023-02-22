#%% 

import numpy as np
import matplotlib.pyplot as plt
from integrate import solve_to
from limit_cycle import isolate_lim_cycle
import unittest
from ODEs import f,g,hopf_normal_form,PPM, VDP
from scipy.integrate import solve_ivp
from integrate import euler_step, RK_step


#%%

result_euler = solve_to(hopf_normal_form, [10,10], 10, 0.0001,method = 'forward_euler')
x_euler  = result_euler.x

result_RK4 = solve_to(hopf_normal_form, [10,10], 10, 0.0001,method = 'ASD')
x_RK4  = result_RK4.x


plt.plot(x_euler[0],x_euler[1])
plt.plot(x_RK4[0],x_RK4[1])

# %%
