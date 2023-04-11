#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from math import floor

#%%

def q(x,u):

    return 1

def gen_diag_mat(N,entries):

    """
    A function that uses scipy.sparse.diags to generated a diagonal matrix

    Parameters
    ----------
    N : Integer
        Size of matrix is NxN
    entries : Python list
        Entries to be placed on the diagonals of the matrix.

    Returns
    -------
    Returns a numpy matrix of size NxN with the entries placed on the diagonals.
    """

    length = len(entries) # Number of diagonals
    lim = floor(length/2) # +/- limits of diagonals
    diagonals = range(-lim,lim+1) # Which diagonals to put the entries in

    k = [[] for _ in range(length)] # Create a list of empty lists
    for i in range(length):
        k[i] = entries[i]*np.ones(N - abs(diagonals[i])) # Fill the lists with the entries
    mat = diags(k,diagonals).toarray() # Create the N-diagonal matrix

    return mat

def f_diff_solver(x_span,b_cond,N,b_cond_type='double-dirichlet'):

    alpha, beta = b_cond # Unpack boundary conditions
    a,b, = x_span # Unpack limits of x
    dx = (b-a)/N 
    x = np.linspace(a,b,N+1)

    if b_cond_type == 'double-dirichlet':

        # alpha, beta = u(a), u(b) respectively

        A_DD = gen_diag_mat(N-1,[1,-2,1])
        b_DD = np.hstack((alpha,np.zeros(N-3),beta))+dx**2*q(x[1:-1],None)
        
        u_sol = np.linalg.solve(A_DD,-b_DD)
        u_full = np.hstack((alpha,u_sol,beta))

    elif b_cond_type == 'dirichlet-neumann':

        # alpha, beta = u(a), u'(b) respectively

        A_DN = gen_diag_mat(N,[1,-2,1])
        A_DN[-1,-2] = 2
        b_DN = np.hstack((alpha,np.zeros(N-2),2*beta*dx))+dx**2*q(x[1:],None)
        
        u_sol = np.linalg.solve(A_DN,-b_DN)
        u_full = np.hstack((alpha,u_sol))

    elif b_cond_type == 'dirichlet-robin':

        # alpha, beta = u(a), u'(b) respectively

        A_DR = gen_diag_mat(N,[1,-2,1])
        A_DR[-1,-2] = 2
        b_DR = np.hstack((alpha,np.zeros(N-2),2*beta*dx))+dx**2*q(x[1:],None)
        
        u_sol = np.linalg.solve(A_DR,-b_DR)
        u_full = np.hstack((alpha,u_sol))

    class result:
        def __init__(self,u,x):
            self.u = u
            self.x = x

    return result(u_full,x)
