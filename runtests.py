
# %%

import numpy as np
import matplotlib.pyplot as plt
from integrate import solve_to
from scipy.integrate import solve_ivp
from ODEs import f,g,VDP,PPM,hopf_normal_form


#%%

# TEST FOR dx_dt = x

def f_scipy(t,x): # Define ODE: dxdt = x, solution: x(t) = x0*exp(x)
    return(x)


x0 = [1]

result = solve_to(f, x0,10,0.01,'forward_euler')
result_true = solve_ivp(f_scipy, [0,10], x0)

x_true = result_true.y
x = result.x
t = result.t_space
t_true = result_true.t

plt.plot(t, x[0])
plt.plot(t_true, x_true[0])
#plt.plot(t,x[0])






#%% 

# TEST FOR PPM MODEL

def PPM_scipy(t,x,a=1,b=0.2,d=0.1): # Define predator-prey model
    x,y = x
    dx_dt = x*(1-x)-(a*x*y)/(d+x)
    dy_dt = b*y*(1-y/x)
    return np.array([dx_dt,dy_dt])


x0 = [10,10]

result = solve_to(PPM, x0,10,0.01,'forward_euler')
result_true = solve_ivp(PPM_scipy, [0,10], x0)

x_true = result_true.y
x = result.x
t = result.t_space


plt.plot(x[0],x[1])
plt.plot(x_true[0],x_true[1])
#plt.plot(t,x[0])

#%%

# TEST FOR HOPF



def hopf_normal_form_scipy(t,x, beta=1,sigma=-1): # Define hopf normal form

    x1,x2 = x
    dx1_dt = beta*x1-x2+sigma*x1*(x1**2+x2**2)
    dx2_dt = x1 + beta*x2 + sigma*x2*(x1**2+x2**2)

    return np.array([dx1_dt,dx2_dt])

x0 = [10,10]

result = solve_to(hopf_normal_form, x0,10,0.01,'forward_euler')
result_true = solve_ivp(hopf_normal_form_scipy, [0,10], x0)

x_true = result_true.y
x = result.x
t = result.t_space


plt.plot(x[0],x[1])
plt.plot(x_true[0],x_true[1])
#plt.plot(t,x[0])



#%%

x0 = [10,10]

result = solve_to(hopf_normal_form, x0,10,0.01,'forward_euler')
result_true = solve_ivp(hopf_normal_form_scipy, [0,10], x0)

x_true = result_true.y
x = result.x
t = result.t_space


plt.plot(x[0],x[1])
plt.plot(x_true[0],x_true[1])
#plt.plot(t,x[0])




