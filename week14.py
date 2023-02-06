#%%
import numpy as np
import matplotlib.pyplot as plt

#%%

def VDP(x,t=0): # Define ODE: Van Der Pol Oscillator
    x_dot = np.zeros(len(x))
    x_dot[0] = x[1]
    x_dot[1] = (1-x[0]**2)*x[1]-x[0]
    return x_dot

def f(x,t=0): # Define ODE: dxdt = x, solution: x(t) = x0*exp(x)
    return(x)

def g(x,t=0): # Define ODE: d2xdt2 = -x
    x_dot=np.zeros(len(x))
    x_dot[0] = x[1]
    x_dot[1] = -x[0]
    return x_dot

def euler_step(deltat_max,x_n, f): # define function to compute a single euler step
    x_n1 = x_n + deltat_max*f(x_n)
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
    t_space = np.arange(0,t+deltat_max,deltat_max)

    # Need two separate versions of solve_to depending on if input is 1-D or not
    # (Could use append but faster to make a single check on an if statement)

    if isinstance(x_old,list): # Case for higher dimensional input (x0 is a list)

        x = np.zeros((len(x_old),len(t_space)))

        if method == 1:  
        # Perform integration of ODE function using EULER METHOD in range t

            for i in t_space:
                x_new = euler_step(deltat_max,x_old,func)
                x_old = x_new
                x[:,counter] = x_new
                counter += 1
            return [x,t_space]

        else: # Perform integration of ODE function using RUNGE-KUTTA method in range t
            
            for i in t_space:
                x_new = RK_step(deltat_max, x_old, i,func)
                x_old = x_new
                x[:,counter] = x_new
                counter += 1
            return [x,t_space]

    else:
        x = np.zeros(len(t_space))
        if method == 1:  
        # Perform integration of ODE function using EULER METHOD in range t

            for i in t_space:
                x_new = euler_step(deltat_max,x_old,func)
                x_old = x_new
                x[counter] = x_new
                counter += 1
            return [x,t_space]

        else: # Perform integration of ODE function using RUNGE-KUTTA method in range t
            
            for i in t_space:
                x_new = RK_step(deltat_max, x_old, i,func)
                x_old = x_new
                x[counter] = x_new
                counter += 1
            return [x,t_space]

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

        


# %% # Produce and plot all results

[error_euler, t_step_space] = error_func(1,0.0000001,0.01, 1)
[error_RK, t_step_space] = error_func(2,0.0000001,0.01, 1)

diff = abs(error_euler-error_RK)
idx = diff == min(diff)
t_step_equal_error = t_step_space[idx] # Time step for which error is equal in both methods

[x_2d,t_space_2d] = solve_to(func = g, x0 = [1,0], t = 100, deltat_max = 0.01, method = 1) # Results for 2-D system in question 3
[x_1d,t_space] = solve_to(func = f, x0 = 1, t = 1, deltat_max = 0.01, method = 1) # Results for first order system in question 1

#%%
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t_space,x_1d)
axs[0, 0].set_title('Solution x(t) = e^t')
axs[0, 0].set(xlabel='t', ylabel='e^t')


axs[0, 1].plot(t_step_space, error_euler, 'tab:blue')
axs[0, 1].plot(t_step_space, error_RK, 'tab:orange')
axs[0, 1].set_title('Error')
axs[0, 1].loglog()
axs[0, 1].legend(loc="upper left")
axs[0, 1].set(xlabel='Time-Step', ylabel='Error')

axs[1, 0].plot(x_2d[0], x_2d[1], 'tab:green')
axs[1, 0].set_title('Phase Portrait')


axs[1, 1].plot(t_space_2d, x_2d[0],  'tab:red')
axs[1, 1].plot(t_space_2d, x_2d[1], 'tab:purple')
axs[1, 1].set_title('Solution for question 3')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# %%
