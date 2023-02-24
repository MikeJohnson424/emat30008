#%%

import numpy as np


def VDP(x,t=0): # Define ODE: Van Der Pol Oscillator
    x1,x2 = x
    x_dot0 = x2
    x_dot1 = (1-x1**2)*x2-x1
    return np.array([x_dot0,x_dot1])

def f(x,t=0): # Define ODE: dxdt = x, solution: x(t) = x0*exp(x)
    return(x)

def g(x,t=0): # Define ODE: d2xdt2 = -x
    x1,x2 = x
    x_dot0 = x2
    x_dot1 = -x1
    return np.array([x_dot0,x_dot1])

def PPM(x,t=0,a=1,b=0.2,d=0.1): # Define predator-prey model
    x,y = x
    dx_dt = x*(1-x)-(a*x*y)/(d+x)
    dy_dt = b*y*(1-y/x)
    return np.array([dx_dt,dy_dt])

def hopf_normal_form(x, t=0, beta=1,sigma=-1): # Define hopf normal form

    x1,x2 = x
    dx1_dt = beta*x1-x2+sigma*x1*(x1**2+x2**2)
    dx2_dt = x1 + beta*x2 + sigma*x2*(x1**2+x2**2)

    return np.array([dx1_dt,dx2_dt])

def hopf_normal_form_sol(t, beta=1, theta=0): # Define solution to hopf normal form

    x1 = np.sqrt(beta)*np.cos(t+theta)
    x2 = np.sqrt(beta)*np.cos(t+theta)

    return np.array([x1,x2])

def third_order_hopf(x, beta=1,sigma=1): # Define a third order hopf normal form ODE

    x1,x2,x3 = x
    dx1_dt = beta*x1-x2+sigma*x1*(x1**2+x2**2)
    dx2_dt = x1 + beta*x2 + sigma*x2*(x1**2+x2**2)
    dx3_dt = -x3

    return np.array([dx1_dt,dx2_dt,dx3_dt])
#%%
















