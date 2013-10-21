import numpy as np
import scipy.linalg as sp
import copy

def generate_matrix(dimension):
    matr = np.matrix(np.zeros((dimension, dimension)))

    for i in range(dimension): # Iteration through the rows
        for j in range(max(0, i-1), dimension): # Iteration through the columns
            matr[i,j] = dimension + 1 - max(i + 1, j + 1) # >indexing from 1
                                                          # >2013
    return matr                                                          

def compute_values(dimension):
    matr = generate_matrix(dimension)

    inf_norm = sp.norm(matr, np.inf)
    one_norm = sp.norm(matr, 1)
    two_norm = sp.norm(matr, 2)
    spec_radius = np.abs(np.max(sp.eigvals(matr)))

    print('Infinity norm: {0:.3e}'.format(inf_norm))
    print('1-norm: {0:.3e}'.format(one_norm))
    print('2-norm: {0:.3e}'.format(two_norm))
    print('Spectral radius: {0:.3e}'.format(spec_radius))

'''
Calculates singular values and eigenvalues and formats them into
tex output
'''
def format_eig_svd():
    def format_cplx(z):
        if z.imag < 1e-300:
            return '{0:.4f}'.format(z.real)
        return '{0:.4f}+{1:.4f}i'.format(z.real, z.imag)

    eig12 = sp.eigvals(generate_matrix(12))
    svd12 = sp.svdvals(generate_matrix(12))

    eig25 = sp.eigvals(generate_matrix(25))
    svd25 = sp.svdvals(generate_matrix(25))

    result12 = r'\begin{tabular}{cc}' + '\n'
    result12 += r'    Eigenvalues&Singular values\\' + '\n'
    result12 += '     \\hline\n'
    result25 = copy.copy(result12)
    for k in range(25):
        if k < 12:
            result12 += r'    ${0}$&${1:.4f}$\\'.format(format_cplx(eig12[k]), svd12[k]) + '\n'
        result25 += r'    ${0}$&${1:.4f}$\\'.format(format_cplx(eig25[k]), svd25[k]) + '\n'

    result12 += '\\end{tabular}\n'
    result25 += '\\end{tabular}\n'

    print(result12)

    print(result25)
