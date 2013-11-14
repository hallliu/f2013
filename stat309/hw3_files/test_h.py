import h_qr
import numpy as np

a = np.array ([[3,2,8],[3,7,3],[12,6,9]], dtype='float64', order='F')
old_a = a.copy()

h_qr.qr_householder.h_qr(a)
q = np.zeros((3,3), dtype='float64', order='F')

q = h_qr.qr_householder.extract_q(a)
a[1:,0] = 0
a[2:,1] = 0
import ipdb;ipdb.set_trace()
