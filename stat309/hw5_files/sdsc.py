import numpy as np
import scipy.sparse as sp

'''
Does steepest descent on A(csr)
'''
def sdsc(A, b, **kwargs):
    x = np.random.randn(b.shape[0])

    iter_th = kwargs.get('iter_th', -1)
    error_th = kwargs.get('error_th', 0)

    iter_ctr = 0
    while True:
        r = b - A*x
        if error_th > 0:
            if np.linalg.norm(r) <= error_th:
                break
        alpha = np.dot(r, r) / np.dot(r, A*r)
        x = x + alpha*r
        iter_ctr += 1
        if iter_ctr == iter_th:
            break

    return (x, iter_ctr)
