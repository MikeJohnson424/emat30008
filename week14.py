
import numpy as np
import matplotlib.pyplot as plt


def f(x,t=0): # Define ODE problem

    x_dot = x
    return(x_dot)


def euler_step(h,x_n, f): # define function to compute a single euler step
    x_n1 = x_n + h*f
    return x_n1

def RK_step(h,x,t,f):
    k1 = f(x,t)
    k2 = f(x+h*k1/2, t+h/2)
    k3 = f(x + h*k2/2, t+ h/2)
    k4 = f(x+h*k3, t+h)
    x_n1 = x + (h/6)*(k1+2*k2+2*k3+k4)
    return x_n1



def solve_to(func, x0, t, deltat_max):

    method = input('Please choose a numerical method: Runge-Kutta [1] Forward Euler [2]')
    t_steps = int(t/deltat_max)
    x_old = x0
    x = np.array([])

    if method == 2:  
        # Perform integration of ODE function using Euler method in range t

        for i in range(t_steps):
            x_new = euler_step(deltat_max,x_old,f(x_old, i*deltat_max))
            x_old = x_new
            x = np.append(x, x_new)
        return [x,t_steps]

    else:
        
        for i in range(t_steps):
            x_new = RK_step(deltat_max, x_old, i*deltat_max,f)
            x_old = x_new
            x = np.append(x,x_new)
        return [x,t_steps]


[x,t_steps] = solve_to(f,1,1,0.01)

#%%

plt.plot(range(t_steps),x)
plt.show()

