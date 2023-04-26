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

1) Define your ODE function, initial conditions, and other parameters if required. For example:

```python
def my_function(x, t, parameters):
    # Your ODE function implementation here
    return x

x0 = [1, 2]  # Initial conditions
t = [0, 10]  # Time range
parameters = []  # Additional parameters for your ODE function
```

2) Choose the integration method ('forward_euler' or 'RK4'), and the maximum time step for integration (deltat_max). For example:

```python
method = 'RK4'
deltat_max = 0.01
```

3) Call the solve_to function with the required parameters:

```python
result = solve_to(my_function, x0, t, parameters, deltat_max, method)
```

4) Access the solution and t_span: 

```python
x_solution = result.x
t_span = result.t_space
```

5) Plot the solution using Matplotlib:

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



### Numerical Continuation

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