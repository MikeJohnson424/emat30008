#%%

import numpy as np
from integrate import solve_to
import matplotlib.pyplot as plt

def f(x,t,args):

    a = args["a"]
    b = args["b"]
    c = args["c"]
    return a*x**2+b*x+c


#%%

result = np.zeros(10)

for i in range(10):

    result[i] = f(5, t = None, args = {"a":i,"b":i,"c":i})

plt.xlabel('parameter')
plt.ylabel('solution')
plt.plot(range(10),result)
plt.show()

# %%

def PPM(x,t=None,args = [1,0.2,0.1]): # Define predator-prey model
    x,y = x
    a,b,d = args
    dx_dt = x*(1-x)-(a*x*y)/(d+x)
    dy_dt = b*y*(1-y/x)
    return np.array([dx_dt,dy_dt])


# %%



idx = 1
args[idx] = alpha


result = PPM([1,2],None,args)
print(result)

# %%
