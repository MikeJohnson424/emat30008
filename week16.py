#%% 

import numpy as np
import matplotlib.pyplot as plt
from integrate import solve_to
import unittest
from functions import f,g,hopf_normal_form,PPM, VDP, hopf_normal_form_sol

result_true = hopf_normal_form_sol

# %%

t_true = np.linspace(0,10,100)

x_true = hopf_normal_form_sol(t_true,1)


result = solve_to(hopf_normal_form,[1,0],10)
x = result.x
t = result.t_space

#%%


plt.plot(t_true,x_true[0])
plt.plot(t,x[0])
# %%
