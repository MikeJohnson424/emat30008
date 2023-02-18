import numpy as np


def VDP(x,t=0): # Define ODE: Van Der Pol Oscillator
    x1,x2 = x
    x_dot0 = x2
    x_dot1 = (1-x1**2)*x2-x1
    return np.hstack((x_dot0,x_dot1))

def f(x,t=0): # Define ODE: dxdt = x, solution: x(t) = x0*exp(x)
    return(x)

def g(x,t=0): # Define ODE: d2xdt2 = -x
    x1,x2 = x
    x_dot0 = x[1]
    x_dot1 = -x[0]
    return np.hstack((x_dot0,x_dot1))

def PPM(x,t,a=1,b=0.2,d=1): # Define predator-prey model
    x,y = x
    dx_dt = x*(1-x)-(a*x*y)/(d+x)
    dy_dt = b*y*(1-y/x)

    return np.array([dx_dt,dy_dt])

def hopf_normal_form(): # Define hopf normal form

def hopf_normal_form_sol(): # Define function solution to hopf normal form
















