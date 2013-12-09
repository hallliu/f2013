import numpy as np
import scipy.sparse as sp
import gepp
import chol
import gsdl
import sdsc
import timeit

def genpossym(n, density, fmt='csr'):
    a = sp.rand(n, n, format='lil', density=density)
    a += a.T
    for i in range(n):
        rowsum = np.abs(a[i, :]).sum() - np.abs(a[i, i])
        a[i, i] = np.abs(rowsum) + 1

    return a.asformat(fmt)

def gendd(n, density, fmt='csr'):
    a = sp.rand(n, n, format='lil', density=density)
    for i in range(n):
        rowsum = np.abs(a[i, :]).sum() - np.abs(a[i, i])
        a[i, i] = np.abs(rowsum) + 1

    return a.asformat(fmt)

'''
Backsolve a sparse A(csr) against b. tri is 'L' if lower-triangular, 'U' otherwise.
'''
def backsolve(_A, tri, b):
    A = sp.csr_matrix(_A)
    x = np.zeros(b.shape[0], dtype=b.dtype)
    it = range(b.shape[0])
    if tri == 'U':
        it = reversed(it)

    for i in it:
        x[i] = (b[i] - A.getrow(i).dot(x)[0]) / A[i, i]
    
    return x

'''
Solve Ax=b using specified method
'''
def gen_solve(A, b, method, **kwargs):
    if method == 'gepp':
        l, u, p = gepp.gepp(A)
        b = b[p]
        y = backsolve(l, 'L', b)
        return backsolve(u, 'U', y)
    
    if method == 'chol':
        f = sp.csr_matrix(chol.chol(A))
        y = backsolve(f, 'L', b)
        return backsolve(f.T, 'U', y)

    if method == 'gsdl':
        return gsdl.gsdl(A, b, **kwargs)

    if method == 'sdsc':
        return sdsc.sdsc(A, b, **kwargs)

def time_accr_solves(n, density):
    '''
    First, run through and compute the results for accuracy.
    '''
    acc = {}
    A = gendd(n, density)
    C = genpossym(n, density)
    b = np.random.randn(n)

    acc['gepp'] = np.linalg.norm(A.dot(gen_solve(A, b, 'gepp')) - b)

    gsdl_data = gen_solve(A, b, 'gsdl', error_th=acc['gepp'], iter_th=100)
    acc['gsdl'] = (np.linalg.norm(A.dot(gsdl_data[0]) - b), gsdl_data[1])

    acc['chol'] = np.linalg.norm(C.dot(gen_solve(C, b, 'chol')) - b)

    sdsc_data = gen_solve(C, b, 'sdsc', error_th=acc['chol'], iter_th=100)
    acc['sdsc'] = (np.linalg.norm(C.dot(sdsc_data[0]) - b), sdsc_data[1])

    print("accuracies done")
    '''
    Now, use the timeit module to time gen_solve for each solution method
    '''
    times = {}
    setup = '''
from main import gen_solve, gendd, genpossym
import numpy as np
A = gendd({0}, {1})
C = genpossym({0}, {1})
b = np.random.randn({0})
'''.format(n, density)
    times['gepp'] = min(timeit.Timer(stmt='gen_solve(A, b, \'gepp\')', setup=setup).repeat(1, 1))
    times['gsdl'] = min(timeit.Timer(stmt='gen_solve(A, b, \'gsdl\', error_th={0}, iter_th=100)'.format(acc['gepp']), setup=setup).repeat(1, 1))
    times['chol'] = min(timeit.Timer(stmt='gen_solve(C, b, \'chol\')', setup=setup).repeat(1, 1))
    times['sdsc'] = min(timeit.Timer(stmt='gen_solve(C, b, \'sdsc\')', setup=setup).repeat(1, 1))

    return (acc, times)
