import numpy as np
import matplotlib.pyplot as plt



def f(x):
    return(x)  # dx/dt = x

def euler_step(h,x_n, f): # define function to compute a single euler step
    x_n1 = x_n + h*f
    return x_n1

def solve_to(func, x0, t, deltat_max):
    # Set up variables/parameters
    t_steps = int(t/deltat_max)
    x_old = x0
    x = np.array([])
    # Calculate x in range t
    for i in range(t_steps):
        x_new = euler_step(deltat_max,x_old,f(x_old))
        x_old = x_new
        x = np.append(x, x_new)
    return [x,t_steps]

[x, t_steps] = solve_to(func = f, x0 = 1, t = 1, deltat_max = 0.01)


plt.plot(range(t_steps),x)
plt.show()