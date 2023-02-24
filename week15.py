
#%%
from integrate import solve_to
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from functions import PPM
from limit_cycle import isolate_lim_cycle

#%% 


ans = isolate_lim_cycle(PPM, [0.5 , 0.3 , 21])

print(ans)
x0 = ans.x0
period = ans.period


x = solve_to(PPM, x0, period).x

plt.plot(x[0],x[1])
#plt.plot(x_true[0],x_true[1])



# %%

