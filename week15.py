
#%%
import numpy as np
import matplotlib.pyplot as plt
from week14 import solve_to
from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp, solve_bvp


#%%
def PPM(x,t=0,a=1,b=0.2,d=0.1):
    x_dot0 = x[0]*(1-x[0])-(a*x[0]*x[1])/(d+x[0])
    x_dot1 = b*x[1]*(1-x[1]/x[0])
    return np.hstack((x_dot0,x_dot1))

[x,t_space] = solve_to(PPM,[0.58,0.27],21,0.01,2)

#plt.plot(t_space,x[0])
plt.plot(x[0],x[1])
plt.show()


#%%

""" def lim_cycle_problem(func, init):

    x0 = init[0:2]
    T = init[-1]
    [x, t_space] = solve_to(func,x0,T)
    sol = x[:,-1]

    dx_dt0 = func(x0)[0]

    return np.hstack((x0-sol, dx_dt0)) """

def lim_cycle_problem(func,init):

    x0 = init[0:-1]
    T = init[-1]
    sol = solve_ivp(func,[0,T],x0).y[:,-1]
    phase_con = func(0, x0)[0]

    return np.hstack((x0-sol,phase_con))



















#%%
"""
def limit_cycle_solver(init, func):

    ans = fsolve(lambda x: lim_cycle_problem(func, x),init)

    sol = ans[0:-1]
    T = ans[-1]

    return [sol, T]



#%%

[sol, T] = limit_cycle_solver([0.58,0.27,25],PPM)

print("Period = " + str(T))
print("Initial condition = " + str(sol))

[x,t_space] = solve_to(PPM,sol,T,0.01,2)


plt.plot(x[0],x[1])
plt.show()


# %%



def BVP(func, x0):
    [x,t_space] = 
 """