import numpy as np
import matplotlib.pyplot as plt

def runtimes_shape(density, results):
    sizes = sorted(results.keys())
    gepp_runtimes = list(map(lambda x: results[x][density][1]['gepp'], sizes))
    sdsc_runtimes = list(map(lambda x: results[x][density][1]['sdsc'], sizes))
    chol_runtimes = list(map(lambda x: results[x][density][1]['chol'], sizes))
    gsdl_runtimes = list(map(lambda x: results[x][density][1]['gsdl'], sizes))

    plt.figure()
    plt.scatter(sizes, gepp_runtimes, marker='.')
    plt.scatter(sizes, sdsc_runtimes, marker='o')
    plt.scatter(sizes, chol_runtimes, marker='*')
    plt.scatter(sizes, gsdl_runtimes, marker='+')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('runtime_shape_{0}.png'.format(int(density*1000)), dpi=200)

def runtimes_density(shape, results):
    ds = sorted(results[shape].keys())
    sdsc_runtimes = list(map(lambda x: results[shape][x][1]['sdsc'], ds))
    gsdl_runtimes = list(map(lambda x: results[shape][x][1]['gsdl'], ds))
    gepp_runtimes = list(map(lambda x: results[shape][x][1]['gepp'], ds))
    chol_runtimes = list(map(lambda x: results[shape][x][1]['chol'], ds))

    plt.figure()
    plt.scatter(ds, sdsc_runtimes, marker='o')
    plt.scatter(ds, gsdl_runtimes, marker='+')
    plt.scatter(ds, gepp_runtimes, marker='.')
    plt.scatter(ds, chol_runtimes, marker='*')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('runtime_density_{0}.png'.format(shape), dpi=200)

def iterations_density(shape, results):
    ds = sorted(results[shape].keys())
    sdsc_iters = list(map(lambda x: results[shape][x][0]['sdsc'][1], ds))
    gsdl_iters = list(map(lambda x: results[shape][x][0]['gsdl'][1], ds))

    plt.figure()
    plt.scatter(ds, sdsc_iters, marker='o')
    plt.scatter(ds, gsdl_iters, marker='+')
    plt.xscale('log')
    plt.savefig('iters_density_{0}.png'.format(shape), dpi=200)

def iterations_shape(density, results):
    sizes = sorted(results.keys())
    sdsc_iters = list(map(lambda x: results[x][density][0]['sdsc'][1], sizes))
    gsdl_iters = list(map(lambda x: results[x][density][0]['gsdl'][1], sizes))

    plt.figure()
    plt.scatter(sizes, sdsc_iters, marker='o')
    plt.scatter(sizes, gsdl_iters, marker='+')
    plt.savefig('iters_shape_{0}.png'.format(int(density*1000)), dpi=200)

def accuracy_shape(density, results):
    sizes = sorted(results.keys())
    gepp_accuracy = list(map(lambda x: results[x][density][0]['gepp'], sizes))
    chol_accuracy = list(map(lambda x: results[x][density][0]['chol'], sizes))

    plt.figure()
    plt.scatter(sizes, gepp_accuracy, marker='.')
    plt.scatter(sizes, chol_accuracy, marker='*')
    plt.ylim(0, max(max(gepp_accuracy), max(chol_accuracy)))
    plt.savefig('acc_shape_{0}.png'.format(int(density*1000)), dpi=200)

def accuracy_density(shape, results):
    ds = sorted(results[shape].keys())
    gepp_accuracy = list(map(lambda x: results[shape][x][0]['gepp'], ds))
    chol_accuracy = list(map(lambda x: results[shape][x][0]['chol'], ds))

    plt.figure()
    plt.scatter(ds, gepp_accuracy, marker='.')
    plt.scatter(ds, chol_accuracy, marker='*')
    plt.ylim(0, max(max(gepp_accuracy), max(chol_accuracy)))
    plt.xscale('log')
    plt.savefig('acc_density_{0}.png'.format(shape), dpi=200)


