import timeit
import cProfile
import numpy as np
from finite_difference import Grid, BoundaryCondition, InitialCondition, diffusion_solver,q

pr = cProfile.Profile()
pr.enable()

grid = Grid(N=10, a=0, b=1)
bc_left = BoundaryCondition('dirichlet', [lambda t: 0],grid)
bc_right = BoundaryCondition('dirichlet', [lambda t: 0],grid)
IC = InitialCondition(lambda x: 10*np.sin(np.pi*x))
t_steps = 100

u = diffusion_solver(grid,
                    bc_left,
                    bc_right,
                    IC,
                    D=0.1,
                    q=q,
                    dt=0.1,
                    t_steps=t_steps,
                    method='lines',
                    storage = 'dense')


pr.disable()
pr.print_stats(sort='cumtime')