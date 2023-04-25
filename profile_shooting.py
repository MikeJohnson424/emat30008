import cProfile, pstats
from BVP import shooting
from scipy.optimize import root
from functions import PPM

def profile_shooting(func,init,parameters,solver):
    # Create a cProfile.Profile object
    pr = cProfile.Profile()

    # Start profiling
    pr.enable()

    # Call the shooting function
    result = shooting(func,init,parameters,solver)

    # Stop profiling
    pr.disable()

    # Create a pstats.Stats object for processing profiling data
    stats = pstats.Stats(pr)

    # Sort and print the profiling results
    stats.strip_dirs().sort_stats('cumulative').print_stats()

    # Return the result from shooting
    return result

profile_shooting(func=PPM,init=[0.5,0.5,20],parameters=[1,0.2,0.1],solver=root)