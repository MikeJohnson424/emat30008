import numpy as np
import matplotlib.pyplot as plt



from week14 import solve_to


def LV(x,a,b,d,t=0):
    x_dot = np.zeros(len(x))
    x_dot[0] = x[0]*(1-x[0])-(a*x[0]*x[1])/(d+x[0])
    x_dot[1] = b*x[1]*(1-x[1]/x[0])
    return x_dot

g = LV([1,5],1,0.3,0.1)

[x,t_space] = solve_to(LV,[1,0])


