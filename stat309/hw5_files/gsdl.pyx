import numpy as np
cimport numpy as np
import scipy.sparse as sp
from libc.stdio cimport printf

'''
Gauss-Seidel, solves Ax=b. Works on csr matrices. Assume that b is a 
plain 1d array, not sparse.
'''
def gsdl(A, np.ndarray[np.float64_t, ndim=1] b, **kwargs):
    cdef np.ndarray[np.float64_t, ndim=1] x
    x = np.random.randn(b.shape[0])

    cdef int iter_counter = 0
    cdef int i
    cdef double A_diag
    cdef int iter_th = kwargs.get('iter_th', -1)
    cdef double error_th = kwargs.get('error_th', 0)
    
    # Objects for storing the data from A
    cdef np.ndarray[np.int32_t, ndim=1] indptrs = A.indptr
    cdef np.ndarray[np.int32_t, ndim=1] indices = A.indices
    cdef np.ndarray[np.float64_t, ndim=1] data = A.data

    cdef np.ndarray[np.int32_t, ndim=1] row_indices
    cdef np.ndarray[np.float64_t, ndim=1] row_data
    cdef int indctr, j
    cdef double asdf

    while True:
        for i in range(x.shape[0]):
            row_indices = indices[indptrs[i]:indptrs[i+1]]
            row_data = data[indptrs[i]:indptrs[i+1]]
            indctr = 0

            x[i] = b[i]
            for j in row_indices:
                # Only do something if we're not on the diagonal.
                # Else, store it so we divide out by it later.
                if j != i:
                    x[i] -= row_data[indctr] * x[j]
                else:
                    A_diag = row_data[indctr]
                indctr += 1
            x[i] /= A_diag
        
        iter_counter += 1

        if iter_counter == iter_th:
            break

        if error_th != 0:
            if iter_counter % 10 == 0:
                asdf = np.linalg.norm(A.dot(x)-b)
                if asdf <= error_th:
                    break
        
    return (x, iter_counter)
