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
    xs = np.empty(svecs.shape[0])
    ys = np.empty(svecs.shape[0])

    for ind in range(svecs.shape[0]):
        xs[ind] = svals[0] * svecs[ind, 0]
        ys[ind] = svals[1] * svecs[ind, 1]

    return (xs, ys)

'''
Makes a labeled plot with names from labels arranged in the proper order
'''
def make_labeled_plot (xs, ys, labels, savename):
    plt.figure(figsize=(8,8))
    plt.scatter(xs, ys)
    for (ind, label) in enumerate(labels):
        plt.annotate(label, xy=(xs[ind], ys[ind]), size='x-small')
    plt.savefig(savename+'.png', format='png', dpi=150, bbox_inches='tight')

def make_overlay_plot(xs1, xs2, ys1, ys2, label1, label2):
    plt.figure(figsize=(8,8))
    plt.axes()
    plt.scatter(xs1, ys1, c='b', marker='.')
    for (ind, label) in enumerate(label1):
        plt.annotate(label, xy=(xs1[ind], ys1[ind]), size='x-small', color='b')

    plt.axes()
    plt.scatter(xs2, ys2, c='g', marker='.')
    for (ind, label) in enumerate(label2):
        plt.annotate(label, xy=(xs2[ind], ys2[ind]), size='x-small', color='g')

    plt.savefig('overlay.png', format='png', dpi=150, bbox_inches='tight')

data = scipy.io.loadmat('processed.mat')['data']
row_names = list(map(lambda x: x.strip(), open('row.txt').readlines()))
col_names = list(map(lambda x: x.strip(), open('column.txt').readlines()))

(u, s, vt) = sl.svd(data)

(xs, ys) = compute_proj_plot(s, u)
make_labeled_plot(xs, ys, row_names, 'countries')

(xs1, ys1) = compute_proj_plot(s, vt.T)
make_labeled_plot(xs1, ys1, col_names, 'features')

make_overlay_plot(xs, xs1, ys, ys1, row_names, col_names)

outlier_points = [row_names.index('Hong Kong'), row_names.index('Singapore'), row_names.index('USA')]
data_no_outliers = data[np.setdiff1d(np.arange(49), outlier_points), :]
(u1, s1, vt1) = sl.svd(data_no_outliers)
(xs2, ys2) = compute_proj_plot(s1, u1)

row_names.remove('Hong Kong')
row_names.remove('Singapore')
row_names.remove('USA')
make_labeled_plot(xs2, ys2, row_names, 'countries_no_outliers')
