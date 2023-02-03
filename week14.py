
import numpy as np
import matplotlib.pyplot as plt


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

    x_old = x0
    x = np.array([])
    t_space = np.arange(0,t,deltat_max)

    if method == 1:  
        # Perform integration of ODE function using EULER METHOD in range t

        for i in t_space:
            x_new = euler_step(deltat_max,x_old,f(x_old, i))
            x_old = x_new
            x = np.append(x, x_new)
        return [x,t_space]

    else: # Perform integration of ODE function using RUNGE-KUTTA method in range t
        
        for i in t_space:
            x_new = RK_step(deltat_max, x_old, i,f)
            x_old = x_new
            x = np.append(x,x_new)
        return [x,t_space]


[x,t_space] = solve_to(f,1,1,0.012,1)

#%%

plt.plot(t_space,x)
plt.show()





