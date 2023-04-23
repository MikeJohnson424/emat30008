#%% 
# test

import unittest
from integrate import solve_to, RK4_step, euler_step
from functions import*
import shooting
import unittest
import numpy as np
from math import isclose
from PDEs import gen_diag_mat, Grid, BoundaryCondition, construct_A_and_b, diffusion_solver, du_dt
from continuation import gen_sol_mat, predictor, corrector,continuation, find_initial_solutions
from scipy.optimize import root
import scipy.sparse as sp

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

    def test_construct_A_and_b(self):
        grid = Grid(5, 0, 1)

        # Test for double dirichlet

        bc_left = BoundaryCondition('dirichlet', [1], grid)
        bc_right = BoundaryCondition('dirichlet', [1], grid)

        A_DD, b_DD = construct_A_and_b(grid,bc_left,bc_right)
        expected_A_DD = np.array([
            [-2,  1,  0,  0],
            [ 1, -2,  1,  0],
            [ 0,  1, -2,  1],
            [ 0,  0,  1, -2]
        ])

        np.testing.assert_array_almost_equal(A_DD,expected_A_DD)

        # Test for dirichlet-neumann

        bc_left = BoundaryCondition('dirichlet', [1], grid)
        bc_right = BoundaryCondition('neumann', [1], grid)

        A_DN, b_DN = construct_A_and_b(grid,bc_left,bc_right)
        expected_A_DN = np.array([
        [-2.,  1.,  0.,  0.,  0.],
        [ 1., -2.,  1.,  0.,  0.],
        [ 0.,  1., -2.,  1.,  0.],
        [ 0.,  0.,  1., -2.,  1.],
        [ 0.,  0.,  0.,  2., -2.]
        ])

        np.testing.assert_array_almost_equal(A_DN,expected_A_DN)
        

        # Test for dirichlet-robin

        bc_left = BoundaryCondition('dirichlet', [1], grid)
        bc_right = BoundaryCondition('robin', [1,2], grid)

        A_DR, b_DR = construct_A_and_b(grid,bc_left,bc_right)
        expected_A_DR = np.array([
        [-2.,  1.,  0.,  0.,  0.],
        [ 1., -2.,  1.,  0.,  0.],
        [ 0.,  1., -2.,  1.,  0.],
        [ 0.,  0.,  1., -2.,  1.],
        [ 0.,  0.,  0.,  2., -2.8]
        ])

        np.testing.assert_array_almost_equal(A_DR,expected_A_DR)



    def test_du_dt(self):

        b = lambda t: 0
        q = lambda x,t,u: 0

        A_dense = np.array([[1, -1], [-1, 1]])
        A_sparse = sp.csr_matrix(np.array([[1, -1], [-1, 1]]))
        A_invalid = "invalid_matrix"
        
        parameters_dense = (A_dense, b, q, 1, 1, 0)
        parameters_sparse = (A_sparse, b, q, 1, 1, 0)
        parameters_invalid = (A_invalid, b, q, 1, 1, 0)

        u = np.array([1, 1])
        t = 0

        # Test du_dt with dense matrix
        result_dense = du_dt(u, t, parameters_dense)
        expected_result_dense = np.array([0, 0])
        np.testing.assert_array_almost_equal(result_dense, expected_result_dense)

        # Test du_dt with sparse matrix
        result_sparse = du_dt(u, t, parameters_sparse)
        expected_result_sparse = np.array([0, 0])
        np.testing.assert_array_almost_equal(result_sparse, expected_result_sparse)

        with self.assertRaises(ValueError):
                du_dt(u, t, parameters_invalid)
    
    def test_gen_sol_mat(self):
        result = gen_sol_mat(3, 4)
        expected_shape = (4, 5)
        self.assertEqual(result.shape, expected_shape)
    
    def test_predictor(self):
        u_current = np.array([1, 2, 3])
        u_old = np.array([0, 1, 2])
        method = 'pArclength'
        step_size = 0.1

        u_pred, delta_u = predictor(u_current, u_old, method, step_size)

        expected_u_pred = np.array([2, 3, 4])
        expected_delta_u = np.array([1, 1, 1])

        np.testing.assert_array_almost_equal(u_pred, expected_u_pred)
        np.testing.assert_array_almost_equal(delta_u, expected_delta_u)
    
    def test_corrector(self):
        pass
    
    def test_find_initial_solutions(self):

        def dummy_myode(x, _, par):
            return par[0] * x - par[1]

        x0 = np.array([1.0])
        par0 = np.array([2.0, 1.0])
        vary_par = 0
        step_size = 1
        solve_for = 'equilibria'

        u_old, u_current = find_initial_solutions(root, dummy_myode, x0, par0, vary_par, step_size, solve_for)

        expected_u_old = np.array([0.5, 2.0])
        expected_u_current = np.array([1/3, 3])

        np.testing.assert_array_almost_equal(u_old, expected_u_old, decimal=6)
        np.testing.assert_array_almost_equal(u_current, expected_u_current, decimal=6)
    
class TestDiffusionSolver(unittest.TestCase):
    
    def test_IVP(self):
        # If initial condition == const.
        # If initial condition == f(x)
        pass
    def test_explicit_euler(self):
        pass
    def test_implicit_euler(self):
        pass
    def test_method_of_lines(self):
        pass
    def test_crank_nicolson(self):
        pass
    # BVP test: constant or function AND dirichlet, neumann, robin etc.
    


      
if __name__ == '__main__':
    unittest.main()


# %%
