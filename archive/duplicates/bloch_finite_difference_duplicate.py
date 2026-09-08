"""Preserve the pre-reorganization duplicate finite-difference Bloch solver.

Method:
    Fourth-order periodic finite-difference first- and second-derivative matrices.
Purpose:
    Retain an auditable duplicate without using it in active workflows.
Inputs:
    Potential samples, periodic cell grid, epsilon, and Bloch momentum.
Outputs:
    Sorted eigenvalues and eigenvectors; no files are written.
Output location:
    None.
Dependencies:
    NumPy and SciPy; no local module imports.
Related files:
    Byte-identical to clr/band_structure/bloch_finite_difference.py before
    organization headers were added.
"""

### DESCRIPTION ###

# solves Bloch eigenvalue problem for the operator - eps^2/2 d/dx^2 + V(x) for a given periodic potential V and k value

### IMPORTS ###

import numpy as np
from scipy.sparse import diags
from scipy.linalg import eig
# import sympy as sym

### FUNCTIONS ###

def bloch(V_per,x_cell,eps,bloch_mom):
    h = x_cell[1]-x_cell[0]
    n_gridpoints = x_cell.size
    e = np.ones(n_gridpoints)
    # construct second derivative matrix
    D2 = h**(-2)*diags( [ (1/2)*(-(5/2)*e) , (4/3)*e[:-1] , -(1/12)*e[:-2] ] , [ 0 , 1 , 2 ] ).toarray()
    D2[0,-2:] += h**(-2)*np.array( [ -(1/12) , (4/3) ] )
    D2[1,-1:] += h**(-2)*np.array( [ -(1/12) ] )
    D2 = D2 + D2.transpose()
    # first derivative matrix
    D1 = h**(-1)*diags( [ (2/3)*e[:-1] , -(1/12)*e[:-2] ] , [ 1 , 2 ] ).toarray()
    D1[0,-2:] += h**(-1)*np.array( [ -(1/12) , (2/3) ] )
    D1[1,-1:] += h**(-1)*np.array( [ -(1/12) ] )
    D1 = D1 - D1.transpose()
    # set up potential matrix
    V_per = diags([V_per],[0]).todense()
    # identity matrix
    I = np.eye(n_gridpoints)
    # assemble Schrodinger operator
    D = (eps**2)*( - (1/2)*D2 - 1j*bloch_mom*D1 + (1/2)*(bloch_mom**2)*I ) + V_per
    # find evalues and evecs
    vals, vecs = eig(D)
    vals = vals.real
    # sort them
    id = np.argsort(vals)
    vals, vecs = vals[id], vecs[:,id]
    return vals, vecs
