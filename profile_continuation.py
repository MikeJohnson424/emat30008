
import cProfile, pstats
from continuation import continuation
from functions import PPM
import numpy as np
from scipy.optimize import root


def profile_continuation(myode, x0, par0, vary_par=0, step_size=0.1, max_steps=50, solve_for='equilibria', method='pArclength', solver=root):
    # Create a cProfile.Profile object
    pr = cProfile.Profile()

    # Start profiling
    pr.enable()

    # Call the continuation function
    result = continuation(myode, x0, par0, vary_par, step_size, max_steps, solve_for, method, solver)

    # Stop profiling
    pr.disable()

    # Create a pstats.Stats object for processing profiling data
    stats = pstats.Stats(pr)

    # Sort and print the profiling results
    stats.strip_dirs().sort_stats('cumulative').print_stats()

    # Return the result from continuation
    return result


x0 = np.array([0.5, 0.5])
par0 = np.array([1, 0.2, 0.1])
vary_par = 0
step_size = 0.1
max_steps = 50
solve_for = 'equilibria'

profiled_result = profile_continuation(PPM, x0, par0, vary_par, step_size, max_steps, solve_for)
print("Solutions:", profiled_result.u)
print("Parameter values:", profiled_result.alpha)
# %%
