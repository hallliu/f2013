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
        f = chol.chol(A)
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

    import ipdb;ipdb.set_trace()
    print('matrices generated')
    #acc['gepp'] = np.linalg.norm(A.dot(gen_solve(A, b, 'gepp')) - b)
    print('gepp done')
    acc['gsdl'] = np.linalg.norm(A.dot(gen_solve(A, b, 'gsdl')[0]) - b)
    print('gsdl done')
    acc['chol'] = np.linalg.norm(A.dot(gen_solve(C, b, 'chol')) - b)
    print('chol done')
    acc['sdsc'] = np.linalg.norm(A.dot(gen_solve(C, b, 'sdsc')) - b)

    print("accuracies done")
    '''
    Now, use the timeit module to time gen_solve for each solution method
    '''
    times = {}
    setup = '''
A = gendd(n, density)
C = genpossym(n, density)
b = np.random.randn(n)
'''
#    times['gepp'] = min(timeit.Timer(stmt='gen_solve(A, b, \'gepp\')', setup=setup).repeat(5, 1))
#    times['gsdl'] = min(timeit.Timer(stmt='gen_solve(A, b, \'gsdl\')', setup=setup).repeat(5, 1))
#    times['chol'] = min(timeit.Timer(stmt='gen_solve(C, b, \'chol\')', setup=setup).repeat(5, 1))
#    times['sdsc'] = min(timeit.Timer(stmt='gen_solve(C, b, \'sdsc\')', setup=setup).repeat(5, 1))

    return (acc, times)
