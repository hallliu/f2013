#!/usr/bin/python
import scipy.io
import scipy.linalg as sl
import numpy as np
import matplotlib.pyplot as plt

'''
Computes the projection of rows of data onto the first two singular vectors,
returns two lists, one with x coords and the other with y coords

Svecs should be a n by 2 numpy array, each column corresponding to the "other" side
singular vector than the one we're projecting onto (see 6a)
'''
def compute_proj_plot(svals, svecs):
    xs = np.empty(data.shape[0])
    ys = np.empty(data.shape[0])

    for ind in range(data.shape[0]):
        xs[ind] = svals[0] * svecs[ind, 0]
        ys[ind] = svals[1] * svecs[ind, 1]

    return (xs, ys)

'''
Makes a labeled plot with names from labels arranged in the proper order
'''
def make_labeled_plot (xs, ys, labels):
    plt.figure()
    plt.plot(xs, ys)
    plt.show()

data = scipy.io.loadmat('processed.mat')
row_names = list(map(lambda x: x.strip, open('row.txt').readlines()))
col_names = list(map(lambda x: x.strip, open('column.txt').readlines()))

(u, s, vt) = sl.svd(data)

(xs, ys) = compute_proj_plot(s, u)
make_labeled_plot(xs, ys)
