from week14 import solve_to
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def PPM(x,t=0,a=1,b=0.2,d=0.1):
    x_dot0 = x[0]*(1-x[0])-(a*x[0]*x[1])/(d+x[0])
    x_dot1 = b*x[1]*(1-x[1]/x[0])
    return np.hstack((x_dot0,x_dot1))
    
def PPM_solve_ivp(t,x,a=1,b=0.2,d=0.1):
    x_dot0 = x[0]*(1-x[0])-(a*x[0]*x[1])/(d+x[0])
    x_dot1 = b*x[1]*(1-x[1]/x[0])
    return np.hstack((x_dot0,x_dot1))

[x, t_space] = solve_to(PPM, [1,2],100)
y = solve_ivp(PPM_solve_ivp,[0,100],[1,2])

plt.plot(x[0],x[1])
plt.plot(y.y[0],y.y[1])
plt.show()


def lim_cycle_problem(func,init):

    x0 = init[0:-1]
    T = init[-1]
    [sol, t_space] = solve_to()





















