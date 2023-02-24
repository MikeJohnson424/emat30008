
#%%

import unittest
from integrate import solve_to
from functions import f,hopf_normal_form,hopf_normal_form_sol


class Test_solve_to(unittest.TestCase):
    
    def test_solve_to(self):
        result1 = solve_to(f,[1],1,0.0001,method='forward_euler').x[0,-1]
        self.assertAlmostEqual(result1,2.718281828459045, places = 3)

        result2 = solve_to(hopf_normal_form,[1,1],1,0.0001,method='forward_euler').x[0,-1]
        self.assertAlmostEqual(result2[0], hopf_normal_form_sol(1)[0])

    def test__RK4_1d(self):
        result = solve_to(f,[1],1,0.1,method='RK4').x[0,-1]
        self.assertAlmostEqual(result, 2.718281828459045, places = 3)


if __name__ == '__main__':
    unittest.main()



# %%
