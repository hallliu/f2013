import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

# These two are the QR algorithms I wrote, imported with f2py
import gram_qr
import h_qr

def compute_errs(n):
    B = np.empty((n, n), dtype='float64', order='F')
    R = np.zeros((n, n), dtype='float64', order='F')
    for i in range(n):
        R[:i+1, i] = np.random.randn(i+1)
        B[:, i] = np.random.randn(n)

    Q = sp.qr(B, mode='full')[0]
    positivize_QR(Q, R)
    A = np.dot(Q, R)

    Rhat_g = A.copy('F')
    Qhat_g = np.empty((n, n), dtype='float64', order='F')

    gram_qr.qr_gram_schmidt.gram_qr(Rhat_g, Qhat_g)

    Rhat_h = A.copy('F')
    Qhat_h = np.zeros((n, n), dtype='float64', order='F')

    h_qr.qr_householder.h_qr(Rhat_h)
    h_qr.qr_householder.extract_q(Rhat_h, Qhat_h)
    for i in range(n):
        Rhat_h[i+1:,i] = 0

    positivize_QR(Qhat_g, Rhat_g)
    positivize_QR(Qhat_h, Rhat_h)
    Ahat_g = np.dot(Qhat_g, Rhat_g)
    Ahat_h = np.dot(Qhat_h, Rhat_h)

    g_Rerr = sp.norm(Rhat_g - R, 'fro') / sp.norm(R, 'fro') 
    h_Rerr = sp.norm(Rhat_h - R, 'fro') / sp.norm(R, 'fro') 

    g_Qerr = sp.norm(Qhat_g - Q, 'fro')
    h_Qerr = sp.norm(Qhat_h - Q, 'fro')

    g_Aerr = sp.norm(Ahat_g - A, 'fro') / sp.norm(A, 'fro') 
    h_Aerr = sp.norm(Ahat_h - A, 'fro') / sp.norm(A, 'fro') 
    import ipdb;ipdb.set_trace()

    return [g_Rerr, g_Qerr, g_Aerr, h_Rerr, h_Qerr, h_Aerr]


# Change signs on Q and R to make all the diagonal entries positive.
# Assumes Q,R are same shape, square
def positivize_QR(Q, R):
    for i in range(Q.shape[0]):
        if R[i, i] < 0:
            R[i, :] *= -1
            Q[:, i] *= -1

def collect_stats():
    sizes = np.concatenate((np.arange(5,100,5), np.arange(100,1000,50), np.arange(1000, 2001, 500)))
    g_Qerrs = np.empty(sizes.shape[0], dtype='float64')
    g_Rerrs = np.empty(sizes.shape[0], dtype='float64')
    g_Aerrs = np.empty(sizes.shape[0], dtype='float64')
    h_Qerrs = np.empty(sizes.shape[0], dtype='float64')
    h_Rerrs = np.empty(sizes.shape[0], dtype='float64')
    h_Aerrs = np.empty(sizes.shape[0], dtype='float64')
    
    for i in range(sizes.shape[0]):
        print("Size {0}".format(sizes[i]))
        g_Rerrs[i], g_Qerrs[i], g_Aerrs[i], h_Rerrs[i], h_Qerrs[i], h_Aerrs[i] = compute_errs(sizes[i])

    return (g_Rerrs, g_Qerrs, g_Aerrs, h_Rerrs, h_Qerrs, h_Aerrs)

def plot_points(g_Rerrs, g_Qerrs, g_Aerrs, h_Rerrs, h_Qerrs, h_Aerrs, sizes):
    plt.figure()
    plt.scatter(sizes, g_Rerrs, c='r', marker='.')
    plt.scatter(sizes, g_Qerrs, c='g', marker='.')
    plt.scatter(sizes, g_Aerrs, c='b', marker='.')
    plt.xscale('log')
    plt.savefig('gram_errs.png', dpi=200, bbox_inches='tight')

    plt.figure()
    plt.scatter(sizes, h_Rerrs, c='r', marker='.')
    plt.scatter(sizes, h_Qerrs, c='g', marker='.')
    plt.scatter(sizes, h_Aerrs, c='b', marker='.')
    plt.xscale('log')
    plt.savefig('householder_errs.png', dpi=200, bbox_inches='tight')

    plt.figure()
    plt.scatter(sizes, g_Aerrs / h_Aerrs, c='g', marker='.')
    plt.xscale('log')
    plt.savefig('backward_errs.png', dpi=200, bbox_inches='tight')

def gen_vandermonde(m, n):
    p = np.empty(m, dtype='float64')
    for i in range(m):
        p[i] = i / (m - 1.)

    A = np.empty((m,n), dtype='float64')
    A[:, 0] = np.ones(m)
    for i in range(1,n):
        A[:, i] = A[:, i - 1] * p

    return A

def solve_Axb_h():
    A = gen_vandermonde(100, 15)
    b = np.arange(100, dtype='float64') / 99
    b = np.exp(np.sin(4*b))

    # Copy A to stick into the Householder QR
    AQ = A.copy('F')
    h_qr.qr_householder.h_qr(AQ)
    
    # Min length soln is given by Rx=c for the first n rows of R, first n entries of c
    h_qr.qr_householder.apply_q_t(AQ, b)

    x = sp.solve_triangular(AQ[:15,:], b[:15])
    return x

def solve_Axb_g():
    A = gen_vandermonde(100, 15)
    b = np.arange(100, dtype='float64') / 99
    b = np.exp(np.sin(4*b))

    Q = np.empty((100, 15), dtype='float64', order='F')
    AC = A.copy('F')
    gram_qr.qr_gram_schmidt.gram_qr(AC, Q)

    c = np.dot(Q.T, b)
    x = sp.solve_triangular(AC[:15, :15], c[:15])
    return x

def solve_aug_h():
    A = gen_vandermonde(100, 15)
    b = np.arange(100, dtype='float64') / 99
    b = np.exp(np.sin(4*b))

    comp = np.bmat([A, np.reshape(b, (100, 1))])
    AQ = comp.copy('F')   
    
    h_qr.qr_householder.h_qr(AQ)
    
    # Min length soln is given by Rx=r for the first n rows/cols of R, r is the last col of R

    x = sp.solve_triangular(AQ[:15,:15], AQ[:15,15])
    return x

def solve_aug_g():
    A = gen_vandermonde(100, 15)
    b = np.arange(100, dtype='float64') / 99
    b = np.exp(np.sin(4*b))

    comp = np.bmat([A, np.reshape(b, (100, 1))])
    Q = np.empty((100, 16), dtype='float64', order='F') # we don't need this shit
    AC = comp.copy('F')
    gram_qr.qr_gram_schmidt.gram_qr(AC, Q)
    x = sp.solve_triangular(AC[:15,:15], AC[:15,15])
    return x

def solve_nml_eqn():
    A = gen_vandermonde(100, 15)
    b = np.arange(100, dtype='float64') / 99
    b = np.exp(np.sin(4*b))
 
    return sp.solve(np.dot(A.T, A), np.dot(A.T, b))

def report_solns():
    print('H Ax=B value: {0}'.format(solve_Axb_h()[14]))
    print('G Ax=B value: {0}'.format(solve_Axb_g()[14]))
    print('H aug value: {0}'.format(solve_aug_h()[14][0]))
    print('G aug value: {0}'.format(solve_aug_g()[14][0]))
    print('nml value: {0}'.format(solve_nml_eqn()[14]))
