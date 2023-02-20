#%%
import numpy as np
import matplotlib.pyplot as plt
from ODEs import f,g


#%%


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

    x = np.zeros([len(x_old),len(t_space)+1])

    if method == 'forward_euler':  # Perform integration of ODE function using EULER METHOD in range t

        for i in t_space:
            x[:,counter] = x_old
            x_new = euler_step(deltat_max,x_old,func)
            x_old = x_new
            counter += 1

        # final iteration where time-step =/= deltat_max:

        delta_t = t - t_space[-1]
        x[:,counter] = euler_step(delta_t, x_old, func)    

    else: # Perform integration of ODE function using RUNGE-KUTTA method in range t
            
        for i in t_space:
            x[:,counter] = x_old
            x_new = RK_step(deltat_max, x_old, i,func)
            x_old = x_new
            counter += 1

        # final iteration where time-step =/= deltat_max:

        delta_t = t - t_space[-1]
        x[:,counter] = RK_step(delta_t, x_old, t,func)

    class result:
        def __init__(self,x,t_space):
            self.x = x
            self.t_space = t_space

    return result(x, np.hstack((t_space, t)))

  
def error_func(deltat_min,deltat_max):

    x_true  = np.exp(1)
    t_step_space = np.logspace(deltat_min,deltat_max,1000)
    error = np.zeros([2,len(t_step_space)])


    for i in [1,2]:

        counter = 0
        for j in t_step_space:
            
            x = solve_to(f,[1],1,j,i).x
            error[i-1,counter] = abs(x[0,-1]-x_true)
            counter +=1

    return [error,t_step_space]

#%%

t = 0.99
deltat_max = 0.1


def gen_t_space(t,deltat_max):

    t_space = np.arange(0,t+deltat_max,deltat_max)
    t_space[-1] = t

    return t_space

t_space_test = gen_t_space(t,deltat_max)

result = solve_to(f,[1],t,deltat_max)

x = result.x
t_space = result.t_space

plt.plot(t_space, x[0])
print(t_space)

# %% # Produce and plot all results

[error, t_step_space] = error_func(10**-5,0.01)

result_2d = solve_to(func = g, x0 = [1,0], t = 100, deltat_max = 0.01) # Results for 2-D system in question 3
result_1d = solve_to(func = f, x0 = [1], t = 1, deltat_max = 0.001) # Results for first order system in question 1

x_1d = result_1d.x
t_space = result_1d.t_space

x_2d = result_2d.x
t_space_2d = result_2d.t_space

#%%

""" fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t_space,x_1d[0])
axs[0, 0].set_title('Solution x(t) = e^t')
axs[0, 0].set(xlabel='t', ylabel='e^t')


axs[0, 1].plot(t_step_space, error[0,:], 'tab:blue')
axs[0, 1].plot(t_step_space, error[1,:], 'tab:orange')
axs[0, 1].set_title('Error')
axs[0, 1].loglog()
axs[0, 1].legend(['Euler','RK4'],loc="best")
axs[0, 1].set(xlabel='Time-Step', ylabel='Error')

axs[1, 0].plot(x_2d[0], x_2d[1], 'tab:green')
axs[1, 0].set_title('Phase Portrait')


axs[1, 1].plot(t_space_2d, x_2d[0],  'tab:red')
axs[1, 1].plot(t_space_2d, x_2d[1], 'tab:purple')
axs[1, 1].set_title('Solution for question 3')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer() """


# %%
