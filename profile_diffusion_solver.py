import cProfile
import pstats
import io
from PDEs import diffusion_solver,Grid,BoundaryCondition
import numpy as np

grid = Grid(100,0,1)
bc_left = BoundaryCondition('dirichlet',[0],grid)
bc_right = BoundaryCondition('dirichlet',[0],grid)
IC = 0
D = 1
q= 0 

def profile_diffusion_solver(grid, bc_left, bc_right, IC, D, q, dt=10, t_steps=20, method='implicit-euler', storage='dense'):
    # Create a cProfile.Profile object
    pr = cProfile.Profile()

    # Start profiling
    pr.enable()

    # Call the diffusion_solver function
    result = diffusion_solver(grid, bc_left, bc_right, IC, D, q, dt, t_steps, method, storage)

    # Stop profiling
    pr.disable()

    # Create a pstats.Stats object for processing profiling data
    stats = pstats.Stats(pr)

    # Sort and print the profiling results
    stats.strip_dirs().sort_stats('cumulative').print_stats()

    # Return the result from diffusion_solver
    return result


profile_diffusion_solver(grid, bc_left, bc_right, IC, D, q=lambda x,t,u: u, dt=10, t_steps=100, method='IMEX', storage='dense')