import cProfile, pstats
from integrate import solve_to
from functions import PPM, hopf_normal_form
import numpy as np


def profile_solve_to(func, x0, t, parameters, deltat_max, method):
    # Create a cProfile.Profile object
    pr = cProfile.Profile()

    # Start profiling
    pr.enable()

    # Call the solve_to function
    result = solve_to(func, x0, t, parameters, deltat_max, method)

    # Stop profiling
    pr.disable()

    # Create a pstats.Stats object for processing profiling data
    stats = pstats.Stats(pr)

    # Sort and print the profiling results
    stats.strip_dirs().sort_stats('cumulative').print_stats()

    # Return the result from solve_to
    return result

profile_solve_to(hopf_normal_form,[1,2],[0,100],[1,-1],0.01,'RK4')
