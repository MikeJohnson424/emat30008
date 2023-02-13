
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

[x,t_space] = solve_to(PPM,[0.4,0.4],100,0.01,2)

#plt.plot(t_space,x[0])
plt.plot(x[0],x[1])
plt.show()



#%%
 
def limit_cycle_solver(init, func):

    def lim_cycle_problem(init):
        x0 = init[0:2]
        T = init[2]
        [x, t_space] = solve_to(func,x0,T)
        x_dot0 = PPM(x0)
        dx_dt0 = x_dot0[0]
        sol = x[:,-1]

        return np.hstack((x[:,0]-sol,dx_dt0))

    ans = fsolve(lim_cycle_problem,init)

    return ans


#%%

ans = limit_cycle_solver([1,0.5,2],PPM)

print(ans)

[x,t_space] = solve_to(PPM,ans[0:2],100,0.01,2)

plt.plot(x[0],x[1])
plt.show()


# %%
