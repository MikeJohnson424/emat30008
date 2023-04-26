# emat30008 - Scientific Computing

Michael Johnson
fr19042@bristol.ac.uk
1962648

This repository contains a collection of functions to solve ordinary differential equations (ODEs) and find limit cycles for given systems. The code provides various numerical integration methods, including the forward Euler method and the Runge-Kutta 4 method. Additionally, it includes functionality for finding the initial conditions and period of limit cycles using the shooting method.

This repository contains code for solving ordinary differential equations (ODEs) using various numerical integration methods, as well as finding limit cycles for given ODE systems.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [ODE integration](#ode-integration)
  - [Numerical Shooting](#numerical-shooting)
  - [Numerical Continuation](#numerical-continuation)
  - [Finite Difference Solvers](#finite-difference-solvers)
- [Testing](#testing)
- [Profiling](#profiling)
- [Report](#report)
- [License](#license)

## Installation

Instructions on how to install and set up the necessary dependencies for the project.

## Usage

### ODE integration

This code is found in `integrate.py` and provides functions to solve Ordinary Differential Equations (ODEs) using the Forward Euler method or the Runge-Kutta 4 (RK4) method. It includes functions `euler_step`, `RK4_step`, and `solve_to`. To use this code, follow the steps below:

1) Import libraries for plotting results/defining ODEs:

```python
import matplotlib.pyplot as plt
import numpy as np
```

2) Define your ODE function, initial conditions, and other parameters if required. For example:

```python
def my_function(x, t, parameters):
    # Your ODE function implementation here
    return x

x0 = [1, 2]  # Initial conditions
t = [0, 10]  # Time range
parameters = []  # Additional parameters for your ODE function
```

3) Choose the integration method ('forward_euler' or 'RK4'), and the maximum time step for integration (deltat_max). For example:

```python
method = 'RK4'
deltat_max = 0.01
```

4) Call the solve_to function with the required parameters:

```python
result = solve_to(my_function, x0, t, parameters, deltat_max, method)
```

5) Access the solution and t_span: 

```python
x_solution = result.x
t_span = result.t_space
```

6) Plot the solution using Matplotlib:

```python
plt.plot(t_span, x_solution[0], label='x1(t)')
plt.plot(t_span_, x_solution[1], label='x2(t)')
plt.xlabel('Time')
plt.ylabel('Solution')
plt.legend()
plt.show()
```

### Numerical shooting

This code is found in `BVP.py` and provides functions to solve for the conditions for a limit cycle of an ODE using the shooting method. It includes functions `lim_cycle_conditions` and `shooting`. The code can be used as follows:

1) Import libraries for defining root solving functions/plotting results:

```python
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt

2) Define your ODE function, initial guess for the limit cycle conditions, and other parameters if required. For example:

```python
def my_function(x, t, parameters):
    # Your ODE function implementation here
    return x

init_guess = [1, 2, 10]  # Initial guess for limit cycle conditions (x0, y0, period)
parameters = []  # Additional parameters for your ODE function
```

3) Call the `shooting` function with the required parameters:

```python
result = shooting(my_function, init_guess, parameters, solver=root)
```

4) Access the limit cycle initial conditions and period:

```python
initial_condition = result.x[0]
period = result.x[2]
```

5) Solve for the limit cycle using a numerical integrator:

```python
solution = solve_ivp(lambda t,x: my_function(x,t,parameters), [0,period], initial_condition)
```

6) Plot the limit cycle using Matplotlib:

```python
plt.plot(solution.y[0],solution.y[1],label = 'Limit Cycle')
```

### Numerical Continuation

Found in `continuation.py` this code provides functions for performing numerical continuation on a function for a given varying parameter. It can be used to track limit cycle conditions or equilibria. Follow the steps below to use this code:

1) Import necessary libraries and functions for plotting and solving: 

```python
from scipy.optimize import solve_ivp
import matplotlib.pyplot as plt
```

2) Define your function, initial guess for the solution, parameters, and the index of the parameter to vary. For example:

```python
def my_function(x, t, parameters):
    # Your ODE function implementation here
    return x

x0 = np.array([1, 1, 10])  # Initial guess for the solution, for example [x0, T] for limit cycle conditions
par0 = np.array([0.5, 2.0, 1.0])  # Additional parameters for your ODE function
vary_par = 0  # Index of the parameter to vary (in this case, the first parameter)
```

3) Set continuation options:

```python
step_size = 0.1
max_steps = 50
solve_for = 'equilibria'  # 'equilibria' or 'limit_cycle'
method = 'pArclength'  # 'pArclength' or 'nParam'
```

4) Call the `continuation` function with the required parameters: 

```python
result = continuation(my_function, x0, par0, vary_par, step_size, max_steps, solve_for, method, solver=root)
```

5) Access the solution and varying parameter values: 

```python
solution_values = result.u
parameter_values = result.alpha
```

6) Optionally, plot solution against varying parameter: 

```python
plt.plot(parameter_values, solution_values[0])
```

### Finite difference solvers



## Testing

This project uses pytest to test the code. To run the tests, type python runtests.py in the terminal.

## Profiling

Users can profile the major functions associated with this package by running 'profile_"function_name".py' files in the terminal. For example, to profile diffusion_solver a user can type:

`python profile_diffusion_solver.py`.

## Report

The jupyter notebook `report.ipynb` contains a report on the project, consisting of a description of the code, examples of usage in addition to a discussion of the major design decisions and a reflective learning log.

## Contributions

## License

Include licensing information for the project.