import numpy as np


def euler_step(deltat_max,x_n, f): # define function to compute a single euler step
    x_n1 = x_n + deltat_max*f(x_n)
    return x_n1

def RK_step(h,x,t,func): # h: time_step, x = current solution, t = current time, f = ODE function
    k1 = func(x,t)
    k2 = func(x+h*k1/2, t+h/2)
    k3 = func(x + h*k2/2, t+ h/2)
    k4 = func(x+h*k3, t+h)
    x_n1 = x + (h/6)*(k1+2*k2+2*k3+k4)
    return x_n1

def solve_to(func, x0, t, deltat_max=0.01, method='RK4'): # Method == 1: EULER, method == 2: RUNGE-KUTTA

    counter = 0
    x_old = np.array(x0)
    t_space = np.arange(0,t+deltat_max,deltat_max)
    t_space[-1] = t

    x = np.zeros([len(x_old),len(t_space)])

    if method == 'forward_euler':  # Perform integration of ODE function using EULER METHOD in range t

        for i in t_space[0:-1]:
            x[:,counter] = x_old
            x_new = euler_step(deltat_max,x_old,f)
            x_old = x_new
            counter += 1

        # final iteration where time-step =/= deltat_max:

        deltat_final = t - t_space[-2]
        x[:,counter] = euler_step(deltat_final, x[:,-2], f)   

    else: # Perform integration of ODE function using RUNGE-KUTTA method in range t
            
        for i in t_space[0:-1]:
            x[:,counter] = x_old
            x_new = RK_step(deltat_max, x_old, i,func)
            x_old = x_new
            counter += 1

        # final iteration where time-step =/= deltat_max:

        delta_t = t - t_space[-2]
        x[:,counter] = RK_step(delta_t, x[:,-2], t,func)

    class result:
        def __init__(self,x,t_space):
            self.x = x
            self.t_space = t_space

    return result(x, t_space)
