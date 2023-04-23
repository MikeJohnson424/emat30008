#%% 
# test

import unittest
from integrate import solve_to, RK4_step, euler_step
from functions import*
import shooting
import unittest
import numpy as np
from math import isclose
from finite_difference import gen_diag_mat, Grid, BoundaryCondition, construct_A_and_b, diffusion_solver
from continuation import gen_sol_mat, predictor, corrector,continuation, find_initial_solutions
from scipy.optimize import root

#%% 

"""
Correctness: Tests should accurately verify that the code performs as expected. 
        This means checking for expected outputs, edge cases, and any possible pitfalls.

Coverage: Aim to achieve high test coverage, which means that a significant portion of the code is being tested. 
        This can be measured using tools like coverage.py. Ideally, tests should cover various scenarios, such as typical use cases, edge cases, and exceptional conditions.

Maintainability: Tests should be easy to maintain as the codebase evolves. 
        They should be modular, well-structured, and follow standard coding practices. 
        Using descriptive names for test functions and including comments can also enhance maintainability.

Readability: Test code should be clear and easy to understand. Following a consistent naming convention, 
        organizing tests logically, and providing clear comments can make tests more accessible for other developers.

Independence: Tests should be independent of each other, meaning the outcome of one test should not affect 
        the outcome of another. This makes it easier to identify the root cause of an issue when a test fails.

Speed: Tests should run quickly to provide fast feedback to developers. This can be achieved by avoiding unnecessary computations,
        using mock objects or stubs when appropriate, and using more efficient testing frameworks.

"""



class TestIntegrationMethods(unittest.TestCase):

    def test_euler_step(self): # Test for euler stepping a function f(x) = x
        x = np.array([1.0])
        t = 0
        deltat_max = 0.1
        expected = np.array([1.1])
        result = euler_step(deltat_max, x, f, [], t)
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_RK4_step(self): # Test for runge-kutta stepping a function f(x) = x
        x = np.array([1.0])
        t = 0
        deltat_max = 0.1
        expected = np.array([1.10517083])
        result = RK4_step(deltat_max, x, f, [], t)
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_solve_to(self): # Test for solving a function f(x) = x to a specified time using both forward euler and fourth order runge-kutta 
        x0 = [1]
        t = 0.4
        deltat_max = 0.0001
        method = 'forward_euler'
        result = solve_to(f, x0, t, [], deltat_max, method)

        self.assertEqual(len(result.t_space), len(result.x[0]), msg="Incorrect dimensions of solution")
        self.assertAlmostEqual(result.t_space[-1], t, msg="Final time in t_space does not match specified time")

        method = 'RK4'
        result = solve_to(f, x0, t, [], deltat_max, method)

        self.assertEqual(len(result.t_space), len(result.x[0]),msg= "Incorrect dimensions of solution")
        self.assertAlmostEqual(result.t_space[-1], t, msg="Final time in t_space does not match specified time")

    def test_solve_to_negative_time_error(self):
        x0 = [1]
        t = -0.4
        deltat_max = 0.0001
        method = 'forward_euler'
        with self.assertRaises(ValueError):
            solve_to(f, x0, t, [], deltat_max, method)


      
if __name__ == '__main__':
    unittest.main()


# %%
