import numpy as np
import scipy.sparse as sp

'''
Does steepest descent on A(csr)
'''
def sdsc(A, b):
    x = np.random.randn(b.shape[0])

    while True:
        r = b - A*x
        if np.allclose(r, 0):
            break
        alpha = np.dot(r, r) / np.dot(r, A*r)
        x = x + alpha*r

    return x
