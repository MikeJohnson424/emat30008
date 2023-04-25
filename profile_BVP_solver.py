import cProfile, pstats
from BVP import BVP_solver
from PDEs import Grid, BoundaryCondition



def profile_BVP_solver(grid,bc_left,bc_right,q,D,u_guess=None):
    # Create a cProfile.Profile object
    pr = cProfile.Profile()

    # Start profiling
    pr.enable()

    # Call the BVP_solver function
    result = BVP_solver(grid,bc_left,bc_right,q,D,u_guess)

    # Stop profiling
    pr.disable()

    # Create a pstats.Stats object for processing profiling data
    stats = pstats.Stats(pr)

    # Sort and print the profiling results
    stats.strip_dirs().sort_stats('cumulative').print_stats()

    # Return the result from BVP_solver
    return result

grid = Grid(100,0,1)
bc_left = BoundaryCondition('dirichlet',[-0],grid)
bc_right = BoundaryCondition('dirichlet',[0],grid)

profile_BVP_solver(grid,bc_left,bc_right,q=1,D=1)