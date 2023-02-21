
#%%
from integrate import solve_to
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from ODEs import PPM
from limit_cyle import isolate_lim_cycle

#%% 


ans = isolate_lim_cycle(PPM, [ 0.5 , 0.3 , 21])

print(ans)
x0 = ans.x0
period = ans.period


x = solve_to(PPM, x0, period).x

plt.plot(x[0],x[1])
#plt.plot(x_true[0],x_true[1])




#%% 
""" def PPM(t,x,a=1,b=0.2,d=0.1): # Define predator-prey model
    x,y = x
    dx_dt = x*(1-x)-(a*x*y)/(d+x)
    dy_dt = b*y*(1-y/x)
    return np.array([dx_dt,dy_dt])

def lim_cycle_problem(func,init):
    
    x0 = init[0:-1]
    T = init[-1]
    a = 0.5
    sol = solve_ivp(func,[0,T],x0).y[:,-1]
    phase_con = func(0,x0)[0]
    phase_con1 = x0[0] - a

    return np.hstack((x0-sol,phase_con))
    

ans = fsolve(lambda x: lim_cycle_problem(PPM, x),[0.5, 0.3, 20])
print(ans) """

# %%

