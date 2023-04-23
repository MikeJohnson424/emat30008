#%% 
# test

import unittest
from integrate import solve_to, RK4_step, euler_step
from functions import*
import shooting
import unittest
import numpy as np
from math import isclose
from PDEs import gen_diag_mat, Grid, BoundaryCondition, construct_A_and_b, diffusion_solver
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
        t = 1
        deltat_max = 0.0001
        expected = np.e

        method = 'forward_euler'
        result = solve_to(f, x0, t, [], deltat_max, method)

        self.assertEqual(len(result.t_space), len(result.x[0]), msg="Time array dimensions do not match solution dimensions")
        self.assertAlmostEqual(result.t_space[-1], t, msg="Final time in t_space does not match time specified by user")
        self.assertAlmostEqual(result.x[0,-1],expected, places = 3,msg="Solution does not match expected solution")

        method = 'RK4'
        result = solve_to(f, x0, t, [], deltat_max, method)

        self.assertEqual(len(result.t_space), len(result.x[0]),msg= "Time array dimensions do not match solution dimensions")
        self.assertAlmostEqual(result.t_space[-1], t, msg="Final time in t_space does not match time specified by user")
        self.assertAlmostEqual(result.x[0,-1],expected, places = 8,msg="Solution does not match expected solution")


    def test_solve_to_errors(self): # Test if solve_to raises correct errors
        x0 = [1]
        t = -0.4
        deltat_max = 0.0001
        method = 'forward_euler'
        with self.assertRaises(ValueError): # Test if solve_to raises error if time is negative
            solve_to(f, x0, t, [], deltat_max, method)
        with self.assertRaises(ValueError):
            solve_to(f, x0, t, [], deltat_max, 'Unrecognised method') # Test if solve_to raises error if method is not recognised

# Class for testing supporting functions to continuation, solve_to and diffusion_solver

class TestComplementaryFunctions(unittest.TestCase): 
    
    def test_construct_A_and_b(self):
        expected = 'an array'
        pass
    
    def test_gen_diag_mat(self):
        expected = np.array([
            [-2,  1,  0,  0],
            [ 1, -2,  1,  0],
            [ 0,  1, -2,  1],
            [ 0,  0,  1, -2]
        ])
        result = gen_diag_mat(4, [1, -2, 1])
        np.testing.assert_array_equal(expected, result)
    
    def test_grid(self):
        grid = Grid(N=10, a=0, b=1)
        self.assertEqual(grid.dx, 0.1)
        np.testing.assert_array_equal(grid.x, np.linspace(0, 1, 11))
        self.assertEqual(grid.left, 0)
        self.assertEqual(grid.right, 1)
    
    def test_boundary_condition(self):
        bcon_type = 'dirichlet'
        value = [1]
        grid = Grid(10, 0, 1)
        bc = BoundaryCondition(bcon_type, value, grid)
        self.assertEqual(bc.type, bcon_type)
        self.assertEqual(bc.value, value)
        self.assertEqual(bc.A_entry, [-2, 1])

        with self.assertRaises(ValueError):
            BoundaryCondition('unrecognised bcon_type', value, grid)
        with self.assertRaises(TypeError):
            BoundaryCondition(bcon_type, 1, grid)
            BoundaryCondition(bcon_type, ['invalid_input'], grid)
    
    def test_gen_sol_mat(self):
        result = gen_sol_mat(3, 4)
        expected_shape = (4, 5)
        self.assertEqual(result.shape, expected_shape)
    
    def test_predictor(self):
        pass
    
    def test_corrector(self):
        pass
    
    def test_find_initial_solutions(self):
        pass
    

    


      
if __name__ == '__main__':
    unittest.main()


# %%
