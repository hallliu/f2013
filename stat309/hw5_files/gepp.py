import numpy as np
import scipy.sparse as sp

'''
Takes a sparse matrix A and returns its LU factorization in dense matrices
'''
def gepp(a):
    l_orig = np.eye(a.shape[0], dtype=a.dtype)
    u_orig = a.A

    for i in range(a.shape[0]):
        l = l_orig[i:, i:]
        u = u_orig[i:, i:]

        l[1:, 0] = -u[1:, 0] / u[0, 0]
        u[1:, 0] = 0
        for j in range(1, u.shape[0]):
            u[j, 1:] = u[j, 1:] + l[j, 0] * u[0, 1:]
    
    return (l_orig, u_orig)
