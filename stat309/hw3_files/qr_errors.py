import numpy as np
import scipy.linalg as sp

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

    Rhat_gram = A.copy('F')
    Qhat_gram = np.empty((n, n), dtype='float64', order='F')

    gram_qr.qr_gram_schmidt.gram_rq(Rhat_gram, Qhat_gram)

    Rhat_h = A.copy('F')
    Qhat_h = np.zeros((n, n), dtype='float64', order='F')

    h_qr.qr_householder.h_qr(Rhat_h)
    h_qr.qr_householder.extract_q(Rhat_h, Qhat_h)
    for i in range(n):
        Rhat_h[i+1:,i] = 0

    positivize_QR(Qhat_gram, Rhat_gram)
    positivize_QR(Qhat_h, Rhat_h)
    Ahat_gram = np.dot(Qhat_gram, Rhat_gram)
    Ahat_h = np.dot(Qhat_h, Rhat_h)

    gram_Rerr = sp.norm(Rhat_gram - R, 'fro') / sp.norm(R, 'fro') 
    h_Rerr = sp.norm(Rhat_h - R, 'fro') / sp.norm(R, 'fro') 

    gram_Qerr = sp.norm(Qhat_gram - Q, 'fro')
    h_Qerr = sp.norm(Qhat_h - Q, 'fro')

    gram_Aerr = sp.norm(Ahat_gram - A, 'fro') / sp.norm(A, 'fro') 
    h_Aerr = sp.norm(Ahat_h - A, 'fro') / sp.norm(A, 'fro') 

    return [gram_Rerr, h_Rerr, gram_Qerr, h_Qerr, gram_Aerr, h_Aerr]


# Change signs on Q and R to make all the diagonal entries positive.
# Assumes Q,R are same shape, square
def positivize_QR(Q, R):
    for i in range(Q.shape[0]):
        if R[i, i] < 0:
            R[i, :] *= -1
            Q[:, i] *= -1
