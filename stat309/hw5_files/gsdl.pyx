import numpy as np
cimport numpy as np
import scipy.sparse as sp

'''
Gauss-Seidel, solves Ax=b. Works on csr matrices. Assume that b is a 
plain 1d array, not sparse.
'''
def gsdl(A, np.ndarray[np.float64_t, ndim=1] b):
    cdef np.ndarray[np.float64_t, ndim=1] x, old_x
    x = np.random.randn(b.shape[0])
    old_x = x.copy()

    cdef int iter_counter = 0
    cdef int i
    cdef double A_diag
    
    # Objects for storing the data from A
    cdef np.ndarray[np.int32_t, ndim=1] indptrs = A.indptr
    cdef np.ndarray[np.int32_t, ndim=1] indices = A.indices
    cdef np.ndarray[np.float64_t, ndim=1] data = A.data

    cdef np.ndarray[np.int32_t, ndim=1] row_indices
    cdef np.ndarray[np.float64_t, ndim=1] row_data
    cdef int indctr, j

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
        if np.allclose(x, old_x):
            break

        old_x = x.copy()

    return (x, iter_counter)
