
#%%
import numpy as np
import matplotlib.pyplot as plt
from week14 import solve_to
from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp, solve_bvp


#%%
def PPM(x,t=0,a=1,b=0.2,d=0.1):
    x_dot = np.zeros(len(x))
    x_dot[0] = x[0]*(1-x[0])-(a*x[0]*x[1])/(d+x[0])
    x_dot[1] = b*x[1]*(1-x[1]/x[0])
    return x_dot

# At b = 0.26 a stable limit cycle is created

[x,t_space] = solve_to(PPM,[1,2,1],100,0.01,2)

plt.plot(x[0],x[1])
plt.show()



#%%
 


def lim_cycle_problem(init):
    x0 = init[0:2]
    [x, t_space] = solve_to(PPM,x0,100)
    x_dot0 = x[1,0]
    sol = x[:,-1]

    return np.hstack((x[:,0]-sol,x_dot0))

ans = fsolve(lim_cycle_problem,[5,2,6])
print(ans)



#%%

def test(x):
    y = 3*x[0]+2*x[1]
    return y

fsolve(test,[2,1])
# %%

