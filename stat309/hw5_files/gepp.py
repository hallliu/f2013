import numpy as np
import scipy.sparse as sp

'''
Takes a sparse matrix A(lil) and returns its LU factorization (both lil)
'''
def gepp(a):
    n = a.shape[0]
    l = sp.identity(n, dtype=a.dtype, format='lil')
    u = sp.lil_matrix(a).copy()

    permut = np.arange(a.shape[0], dtype='int32')

    for i in range(a.shape[0] - 1):

        # Find max leading entry
        maxind = np.argmax(np.abs(u[i:, i]).A)

        # Swap them in the thing we're using to keep track of permutations
        i1 = np.where(permut == maxind + i)
        i2 = np.where(permut == i)
        temp = permut[i1]
        permut[i1] = permut[i2]
        permut[i2] = temp

        # Actually swap the rows
        temp = u[i, :]
        u[i, :] = u[maxind + i, :]
        u[maxind + i, :] = temp

        # Swap the rows on the L matrix too
        if i != 0:
            temp = l[i, :i]
            l[i, :i] = l[maxind + i, :i]
            l[maxind + i, :i] = temp

        # Do Gaussian elimination
        l[i+1:, i] = u[i+1:, i] / u[i, i]
        u[i+1:, i] = 0
        for j in range(i+1, u.shape[1]):
            u[i+1:, j] = u[i+1:, j] - l[i+1:, i] * u[i, j]

    return (l, u, permut)
