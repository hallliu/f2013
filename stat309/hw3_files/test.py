import gram_qr
import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype='float64', order='F')
q = np.empty(a.shape, dtype='float64', order='F')

gram_qr.qr_gram_schmidt.gram_qr(a,q)

b = np.random.random((40,50)).T
old_b = b.copy()
q1 = np.empty((50,50), dtype='float64', order='F')

gram_qr.qr_gram_schmidt.gram_qr(b,q1)
import ipdb;ipdb.set_trace()
