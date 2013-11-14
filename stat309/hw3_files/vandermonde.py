import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

# These two are the QR algorithms I wrote, imported with f2py
import gram_qr
import h_qr

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
