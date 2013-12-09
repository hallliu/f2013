import numpy as np
import scipy.sparse as sp

'''
Sparse Cholesky factorization (on lil matrices)
'''
def chol(a):
    f = sp.lil_matrix(a.shape, dtype=a.dtype)
    a_copy = sp.lil_matrix(a).copy()
    for i in range(a.shape[0]):
        f[i:, i] = a_copy[i:, i] / np.sqrt(a_copy[i, i])
        a_copy[i+1:, i+1:] -= f[i+1:, i] * f[i+1:, i].T

    return f


