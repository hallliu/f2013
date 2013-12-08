import numpy as np
import scipy.sparse as sp

'''
Takes a sparse matrix A and returns its LU factorization in dense matrices
'''
def gepp(a):
    l_orig = np.eye(a.shape[0], dtype=a.dtype)
    u_orig = a.A

    permut = np.arange(a.shape[0], dtype='int32')

    for i in range(a.shape[0]):
        l = l_orig[i:, i:]
        u = u_orig[i:, i:]

        # Find max leading entry
        maxind = np.argmax(np.abs(u[:, 0]))
        # Swap them in the thing we're using to keep track of permutations
        i1 = np.where(permut == maxind + i)
        i2 = np.where(permut == i)
        temp = permut[i1]
        permut[i1] = permut[i2]
        permut[i2] = temp

        # Actually swap the rows
        temp = u[0, :].copy()
        u[0, :] = u[maxind, :]
        u[maxind, :] = temp

        # Swap the rows on the L matrix too
        temp = l_orig[i, :i].copy()
        l_orig[i, :i] = l_orig[maxind + i, :i]
        l_orig[maxind + i, :i] = temp

        # Do Gaussian elimination
        l[1:, 0] = u[1:, 0] / u[0, 0]
        u[1:, 0] = 0
        for j in range(1, u.shape[1]):
            u[1:, j] = u[1:, j] - l[1:, 0] * u[0, j]

    
    return (l_orig, u_orig, permut)
