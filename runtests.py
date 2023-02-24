
# %%

import numpy as np
import matplotlib.pyplot as plt
from integrate import solve_to, RK4_step
from scipy.integrate import solve_ivp
from ODEs import f,g,VDP,PPM,hopf_normal_form, hopf_normal_form_sol


#%%

# TEST FOR dx_dt = x

def f_scipy(t,x): # Define ODE: dxdt = x, solution: x(t) = x0*exp(x)
    return(x)


x0 = [1]

result = solve_to(f, x0,10,0.01)
result_true = solve_ivp(f_scipy, [0,10], x0)

x_true = result_true.y
x = result.x
t = result.t_space
t_true = result_true.t

plt.plot(t, x[0])
plt.plot(t_true, x_true[0])
#plt.plot(t,x[0])


#%% 

# TEST FOR VDP OSCILLATOR

def VDP_scipy(t,x): # Define ODE: Van Der Pol Oscillator
    x1,x2 = x
    x_dot0 = x2
    x_dot1 = (1-x1**2)*x2-x1
    return np.array([x_dot0,x_dot1])

x0 = [1,1]

result = solve_to(VDP, x0,10,0.01)
result_true = solve_ivp(VDP_scipy, [0,10], x0)

x_true = result_true.y
x = result.x
t = result.t_space


plt.plot(x[0],x[1])
plt.plot(x_true[0],x_true[1])
#plt.plot(t,x[0])

#%% 

# TEST FOR PPM MODEL

def PPM_scipy(t,x,a=1,b=0.2,d=0.1): # Define predator-prey model
    x,y = x
    dx_dt = x*(1-x)-(a*x*y)/(d+x)
    dy_dt = b*y*(1-y/x)
    return np.array([dx_dt,dy_dt])


x0 = [1,10]

result = solve_to(PPM, x0,10,0.01)
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

x0 = [1,1]

result = solve_to(hopf_normal_form, x0,10,0.1)
result_true = solve_ivp(hopf_normal_form_scipy, [0,10], x0)

x_true = result_true.y
x = result.x
t = result.t_space



plt.plot(x[0],x[1])
plt.plot(x_true[0],x_true[1])

#plt.plot(t,x[0])


# %%

# TESTING OLD CODE

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

def solve_to(func, x0, t, deltat_max=0.01, method=2): # Method == 1: EULER, method == 2: RUNGE-KUTTA

    counter = 0
    x_old = np.array(x0)
    t_space = np.arange(0,t+deltat_max,deltat_max)


    x = np.zeros([len(x_old),len(t_space)])

    if method == 1:  
    # Perform integration of ODE function using EULER METHOD in range t

        for i in t_space:
            x[:,counter] = x_old
            x_new = euler_step(deltat_max,x_old,func)
            x_old = x_new
            counter += 1
    

    else: # Perform integration of ODE function using RUNGE-KUTTA method in range t
            
        for i in t_space:
            x[:,counter] = x_old
            x_new = RK_step(deltat_max, x_old, i,func)
            x_old = x_new
            counter += 1

    return [x,t_space]

def hopf_normal_form_scipy(t,x, beta=1,sigma=-1): # Define hopf normal form

    x1,x2 = x
    dx1_dt = beta*x1-x2+sigma*x1*(x1**2+x2**2)
    dx2_dt = x1 + beta*x2 + sigma*x2*(x1**2+x2**2)

    return np.array([dx1_dt,dx2_dt])

x0 = [1,1]

[x, t_space] = solve_to(hopf_normal_form, x0,10,0.001,'runge_kutta')
result_true = solve_ivp(hopf_normal_form_scipy, [0,10], x0)

x_true = result_true.y


plt.plot(x[0],x[1])
plt.plot(x_true[0],x_true[1])
#plt.plot(t,x[0])

# %%
