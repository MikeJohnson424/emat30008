#%%
import numpy as np
import matplotlib.pyplot as plt
from ODEs import f,g
from integrate import solve_to


#%%

  
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

t = 1.01
deltat_max = 0.01


def gen_t_space(t,deltat_max):

    t_space = np.arange(0,t+deltat_max,deltat_max)
    t_space[-1] = t

    return t_space

t_space_test = gen_t_space(t,deltat_max)

result = solve_to(f,[1],t,deltat_max, method='runge')

x = result.x
t_space = result.t_space

plt.plot(t_space, x[0])
print(t_space)
print(len(t_space))

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
