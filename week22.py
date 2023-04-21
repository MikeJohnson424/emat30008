#%%

import numpy as  np
from week19 import construct_A_and_b, BoundaryCondition, Grid 
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from week20 import InitialCondition
import scipy.sparse as sp


u_new = np.linalg.solve(eye(len(A))-C*A,u_old+dt*q)





