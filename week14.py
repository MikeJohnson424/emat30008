#%%


import numpy as np
import matplotlib.pyplot as plt

#%% 
def f(x,t=0): # Define ODE problem

    x_dot = x
    return(x_dot)


def euler_step(h,x_n, f): # define function to compute a single euler step
    x_n1 = x_n + h*f
    return x_n1

def RK_step(h,x,t,f): # h: time_step, x = current solution, t = current time, f = ODE function
    k1 = f(x,t)
    k2 = f(x+h*k1/2, t+h/2)
    k3 = f(x + h*k2/2, t+ h/2)
    k4 = f(x+h*k3, t+h)
    x_n1 = x + (h/6)*(k1+2*k2+2*k3+k4)
    return x_n1

def solve_to(func, x0, t, deltat_max, method): # Method == 1: EULER, method == 2: RUNGE-KUTTA

    counter = 0
    x_old = x0
    t_space = np.arange(0,t,deltat_max)
    x = np.zeros(len(t_space))

    if method == 1:  
        # Perform integration of ODE function using EULER METHOD in range t

        for i in t_space:
            x_new = euler_step(deltat_max,x_old,f(x_old, i))
            x_old = x_new
            x[counter] = x_new
            counter += 1
        return [x,t_space]

    else: # Perform integration of ODE function using RUNGE-KUTTA method in range t
        
        for i in t_space:
            x_new = RK_step(deltat_max, x_old, i,f)
            x_old = x_new
            x[counter] = x_new
            counter += 1
        return [x,t_space]


""" [x,t_space] = solve_to(f,1,1,0.01,2)

plt.plot(t_space,x)
plt.show() """

#%%

def error_func(method,deltat_min, deltat_max, t): # Method = 1 or 2 = euler or RK, delta t min and max is range of time-steps to calculate error for, t is time at which we want solution

    counter = 0
    x_true = np.exp(t)
    t_step_space = np.linspace(deltat_min,deltat_max,1000)
    error = np.zeros(len(t_step_space))

    for i in t_step_space:

        [x, t_space] = solve_to(f,1,t,i,method)
        error[counter] = abs(x[-1]-x_true)
        counter +=1

    return [error, t_step_space]

[error, t_step_space] = error_func(2,0.1,0.00001, 1)

plt.xscale("log")
plt.yscale("log")
plt.plot(t_step_space,error)
plt.show()
        





# %%
