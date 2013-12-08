import numpy as np
import scipy.sparse as sp

'''
Dense Cholesky factorization
'''
def chol(a):
    f = np.zeros(a.shape, dtype=a.dtype)
    a_copy = a.copy()
    for i in range(a.shape[0]):
        f[i:, i] = a_copy[i:, i] / np.sqrt(a_copy[i, i])
        a_copy[i+1:, i+1:] -= np.outer(f[i+1:, i], f[i+1:, i])

    return f

def genpossym(n):
    a = np.random.randn(n, n)
    a += a.T
    for i in range(n):
        rowsum = np.sum(np.abs(a[i, :])) - np.abs(a[i, i])
        a[i, i] = np.abs(rowsum) + 1

    return a
