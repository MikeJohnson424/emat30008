
#%%
from integrate import solve_to
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from scipy.integrate import solve_ivp
from functions import PPM
from shooting import shooting

#%%

init = [0.57,0.31,21]; func = PPM; parameters = [1,0.2,0.1]

ans = shooting(func,init,parameters)

ans = root(lambda x: shooting(func,x,parameters),init)
print(ans.x)


sol = solve_to(PPM,x0,T,[1,0.2,0.1])
plt.plot(sol.x[0],sol.x[1])

# %%

