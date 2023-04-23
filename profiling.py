import timeit
import cProfile
import numpy as np
from PDEs import Grid, BoundaryCondition, diffusion_solver

pr = cProfile.Profile()
pr.enable()

grid = Grid(N=10, a=0, b=1)
bc_left = BoundaryCondition('dirichlet', [lambda t: 0],grid)
bc_right = BoundaryCondition('dirichlet', [lambda t: 0],grid)
t_steps = 100

u = diffusion_solver(grid,
                    bc_left,
                    bc_right,
                    IC=0,
                    D=0.1,
                    q=1,
                    dt=0.1,
                    t_steps=t_steps,
                    method='lines',
                    storage = 'sparse')


pr.disable()
pr.print_stats(sort='cumtime')