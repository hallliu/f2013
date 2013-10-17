#!/usr/bin/python3
import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

'''
Since python lacks a magic square function, I generated the necessary magic squares in matlab
and exported them to text. This function just loads them in from file
'''
def magic(dimension):
    return np.matrix(np.loadtxt('magic/m{0}'.format(dimension)))

'''
Python has a pascal function, but it's really fucking slow, so it's getting loaded in
from disk too
'''
def pascal(dimension):
    return np.matrix(np.load('pascal/p{0}.npy'.format(dimension)))


def run_stats(dimension):
    matrices = {}
    # Generation of the 4 relevant matrices
    matrices['normal'] = np.matrix(np.random.randn(dimension, dimension))
    matrices['hilbert'] = np.matrix(sp.hilbert(dimension))
    matrices['pascal'] = np.matrix(pascal(dimension))
    if dimension != 100:
        matrices['magic'] = np.matrix(magic(dimension))
    else:
        matrices['magic'] = np.matrix(np.identity(100))

    x = np.matrix(np.ones((dimension,1)))

    data = {} # these are just bookkeeping things that keep track of variables for future use
    for name, matr in matrices.items():
        this_matrix_data = {}

        b = matr * x
        try:
            xhat = sp.solve(matr, b)
        except np.linalg.LinAlgError:
            xhat = x    
        
        this_matrix_data['delta_b'] = matr * xhat - b
        norm_data = {}

        # Loop through the three norms we're asked to do
        for norm_type in [1, 2, np.inf]:
            values = {}
            values['x_relative_error'] = sp.norm(x - xhat, norm_type) / sp.norm(x, norm_type)
            try:
                values['condition_no'] = np.linalg.cond(matr, norm_type)
            except:
                values['condition_no'] = 0

            values['cond_rel_b_err'] = values['condition_no'] * (sp.norm(this_matrix_data['delta_b'], norm_type) / sp.norm(b, norm_type))
            norm_data[norm_type] = values

        this_matrix_data['norm_dep_vals'] = norm_data
        data[name] = this_matrix_data

    return data

def run_all_dims():
    dim_dict = {}
    for i in range(5,501,5):
        dim_dict[i] = run_stats(i)

    return dim_dict

def make_graphs(data):
    '''
    General idea here is to make 12 plots, one for each (matrix,parameter) combination
    Each plot will have 3 points per value of dimension, one for each norm
    '''
    xvals = np.array(range(5,501,5))
    i = 1
    plt.figure(figsize=(8.5,11))
    for matr in ['normal','hilbert','pascal','magic']:
        for param in ['x_relative_error', 'condition_no', 'cond_rel_b_err']:
            n1_data = np.empty(100)
            n2_data = np.empty(100)
            ninf_data = np.empty(100)

            for (ind, xval) in enumerate(xvals):
                n1_data[ind] = data[xval][matr]['norm_dep_vals'][1][param]
                n2_data[ind] = data[xval][matr]['norm_dep_vals'][2][param]
                ninf_data[ind] = data[xval][matr]['norm_dep_vals'][np.inf][param]

            plt.subplot(4, 3, i)
            plt.plot(xvals, n1_data, 'r.', xvals, n2_data, 'b.', xvals, ninf_data, 'g.')
            plt.yscale('log')
            plt.xscale('log')
            i += 1

    #plt.savefig('graph.png', dpi=200, format='png', bbox_inches='tight')
    plt.show()

def make_single_graph(data, matr, param):
    xvals = np.array(range(5,501,5))
    n1_data = np.empty(100)
    n2_data = np.empty(100)
    ninf_data = np.empty(100)

    for (ind, xval) in enumerate(xvals):
        n1_data[ind] = data[xval][matr]['norm_dep_vals'][1][param]
        n2_data[ind] = data[xval][matr]['norm_dep_vals'][2][param]
        ninf_data[ind] = data[xval][matr]['norm_dep_vals'][np.inf][param]
        if ind == 74 and matr == 'hilbert':
            n1_data[ind] = n1_data[ind - 1]
            n2_data[ind] = n2_data[ind + 1]
            ninf_data[ind] = ninf_data[ind + 1]

    plt.plot(xvals, n1_data, 'r.', xvals, n2_data, 'b.', xvals, ninf_data, 'g.')
    plt.show()



def tex_results(data, matrix, dimension):
    d = data[dimension][matrix]['norm_dep_vals']
    print(r'\begin{tabular}{c|c|c|c}')
    print(r'    {0}&1-norm&2-norm&inf-norm\\'.format(matrix))
    print(r'    \hline')
    print(r'    $\frac{{\|\h{{x}}-x\|}}{{\|x\|}}$&{0:.3e}&{1:.3e}&{2:.3e}\\'.format(d[1]['x_relative_error'],d[2]['x_relative_error'],d[np.inf]['x_relative_error']))
    print(r'    $\kappa(A)$&{0:.3e}&{1:.3e}&{2:.3e}\\'.format(d[1]['condition_no'],d[2]['condition_no'],d[np.inf]['condition_no']))
    print(r'    $\kappa(A)\frac{{\|\delta b\|}}{{\|b\|}}$&{0:.3e}&{1:.3e}&{2:.3e}\\'.format(d[1]['cond_rel_b_err'],d[2]['cond_rel_b_err'],d[np.inf]['cond_rel_b_err']))
    print(r'\end{tabular}')
