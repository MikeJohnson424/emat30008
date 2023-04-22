#%% 
# test

import unittest
import integrate
from functions import f,hopf_normal_form,hopf_normal_form_sol, PPM
import shooting

#%% 

class Test_solve_to(unittest.TestCase):
    
    def test_solve_to(self):

        # Test solver against known value of e using forward euler

        result1 = integrate.solve_to(f,[1],1,0.0001,method='forward_euler').x[0,-1]
        self.assertAlmostEqual(result1,2.718281828459045, places = 3)

        # Test solver against known solution of hopf normal form

        result2 = integrate.solve_to(hopf_normal_form,[1,0],1)
        result2_true = hopf_normal_form_sol(1)[0]
        self.assertAlmostEqual(result2.x[0,-1], result2_true, places = 3)

        # Test solver against known value of e using Runge-Kutta 4 method

        result3 = integrate.solve_to(f,[1],1,0.01,method='RK4').x[0,-1]
        self.assertAlmostEqual(result3,2.718281828459045, places = 3)

        # Test solver on a third order hopf normal form

        # Test for incorrect dimensions

        

    def test_isolate_lim_cycle(self):

        result = shooting.isolate_lim_cycle(PPM, [0.5 , 0.3 , 21])
        self.assertAlmostEqual(result.T, 20.8168665840617, places = 2)
        
if __name__ == '__main__':
    unittest.main()


# %%
