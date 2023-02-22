
#%%

import numpy as np
import matplotlib.pyplot as plt

from integrate import solve_to
from limit_cyle import isolate_lim_cycle
from ODEs import third_order_hopf, hopf_normal_form, hopf_normal_form_sol

from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp, solve_bvp


#%%

result = solve_to(hopf_normal_form,[1,2],100,0.01)
x = result.x
plt.plot(x[0],x[1])

# %%

def hopf_normal_form(t,x, beta=1,sigma=-1): # Define hopf normal form

    x1,x2 = x
    dx1_dt = beta*x1-x2+sigma*x1*(x1**2+x2**2)
    dx2_dt = x1 + beta*x2 + sigma*x2*(x1**2+x2**2)
    return np.array([dx1_dt,dx2_dt])

result = solve_ivp(hopf_normal_form,[0,10],[1,2])
x = result.y
plt.plot(x[0],x[1])

# %%
