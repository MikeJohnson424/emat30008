
import numpy as np


""" == ODE FUNCTIONS == """

def VDP(x,t=None,parameters=[]): # Define ODE: Van Der Pol Oscillator
    x1,x2 = x
    x_dot0 = x2
    x_dot1 = (1-x1**2)*x2-x1
    return np.array([x_dot0,x_dot1])

def f(x,t=None,parameters=[]): # Define ODE: dxdt = x, solution: x(t) = x0*exp(x)
    return(x)

def g(x,t=None,parameters=[]): # Define ODE: d2xdt2 = -x
    x1,x2 = x
    x_dot0 = x2
    x_dot1 = -x1
    return np.array([x_dot0,x_dot1])

def PPM(x,t=None,parameters = [1,0.2,0.1]): # Define predator-prey model

    a,b,d = parameters
    x,y = x
    dx_dt = x*(1-x)-(a*x*y)/(d+x)
    dy_dt = b*y*(1-y/x)
    return np.array([dx_dt,dy_dt])


def hopf_normal_form(x, t=None, parameters = [1,-1]): # Define hopf normal form
    beta, sigma = parameters
    x1,x2 = x
    dx1_dt = beta*x1-x2+sigma*x1*(x1**2+x2**2)
    dx2_dt = x1 + beta*x2 + sigma*x2*(x1**2+x2**2)

    return np.array([dx1_dt,dx2_dt])

def third_order_hopf(x, t=None, parameters = [1,-1]): # Define a third order hopf normal form ODE
    beta, sigma = parameters
    x1,x2,x3 = x
    dx1_dt = beta*x1-x2+sigma*x1*(x1**2+x2**2)
    dx2_dt = x1 + beta*x2 + sigma*x2*(x1**2+x2**2)
    dx3_dt = -x3

    return np.array([dx1_dt,dx2_dt,dx3_dt])

def modified_hopf(x, t=None, parameters = [1]):
    
    beta = parameters
    x1,x2 = x
    dx1_dt = beta*x1 - x2 + x1*(x1**2 + x2**2) - x1*(x1**2 + x2**2)**2
    dx2_dt = x1 + beta*x2 + x2*(x1**2 + x2**2) - x2*(x1**2 + x2**2)**2

    return np.array([dx1_dt, dx2_dt])


def diffusion_ODE(x,t=None,parameters = []):
    x1,x2 = x
    D,q = parameters
    dx1_dt = x2
    dx2_dt = -q/D

    return np.array([dx1_dt,dx2_dt])

""" == OTHER FUNCTIONS == """

def hopf_normal_form_sol(t, beta=1, theta=0): # Define solution to hopf normal form

    x1 = np.sqrt(beta)*np.cos(t+theta)
    x2 = np.sqrt(beta)*np.sin(t+theta)

    return np.array([x1,x2])

def h(x,t=None,parameters=[]):
    return x[0]**3-x[0]+parameters[0]

def sol_source(x,a,b,alpha,beta,D): # True solution to week19 question 2

    return -1/(2*D)*(x-a)*(x-b)+(beta-alpha)/(b-a)*(x-a)+alpha

def sol_no_source(x,a,b,alpha,beta): # True solution to week 19 question 1

    return ((beta-alpha))/(b-a)*(x-a)+alpha

def bratu(u,t,parameters):

    mu,D,dx,A = parameters
    
    return D*np.matmul(A,u) + dx**2*np.exp(mu*u)


















