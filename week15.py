
#%%
import numpy as np
import matplotlib.pyplot as plt
from week14 import solve_to
from scipy.optimize import fsolve
from scipy.integrate import odeint


#%%
def PPM(x,t=0,a=1,b=0.26,d=0.1):
    x_dot = np.zeros(len(x))
    x_dot[0] = x[0]*(1-x[0])-(a*x[0]*x[1])/(d+x[0])
    x_dot[1] = b*x[1]*(1-x[1]/x[0])
    return x_dot

# At b = 0.26 a stable limit cycle is created

[x,t_space] = solve_to(PPM,[0.5,0.1],100,0.001,2)

#scipy_results = odeint(PPM,[0.5,0.1],t_space)

#plt.plot(x[0], x[1])
#plt.plot(scipy_results[:,0],scipy_results[:,1])
plt.plot(t_space,x[0].T)
#plt.plot(t_space,x[1].T)
plt.show()

fsolve(PPM, [2,0])





# %%
