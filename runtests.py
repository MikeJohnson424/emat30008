#%% 

import numpy as np
import matplotlib.pyplot as plt
from integrate import solve_to
from limit_cyle import isolate_lim_cycle
import unittest
from ODEs import f,g,hopf_normal_form,PPM, VDP
from scipy.integrate import solve_ivp

#%%

def hopf_normal_form_scipy(t,x, beta=1,sigma=-1): # Define hopf normal form

    x1,x2 = x
    dx1_dt = beta*x1-x2+sigma*x1*(x1**2+x2**2)
    dx2_dt = x1 + beta*x2 + sigma*x2*(x1**2+x2**2)

    return np.array([dx1_dt,dx2_dt])
#%%

x0 = [10,10]

result = solve_to(hopf_normal_form, x0,10)
result_true = solve_ivp(hopf_normal_form_scipy, [0,10], x0)

x_true = result_true.y
x = result.x
t = result.t_space


plt.plot(x[0],x[1])
plt.plot(x_true[0],x_true[1])
#plt.plot(t,x[0])

# %%
