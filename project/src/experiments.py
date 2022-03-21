from collections import defaultdict
import cirq
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import timeit

from algorithms import get_sampling_algorithm
from circuitUtils import (get_sycamore23_qubits, get_sycamore_circuit,
                          sample_circuit)
from sampling_algorithm import SamplingAlgorithm


"""Examples of Quantum Simulations

This script allows the user to try several ways in which to sample
different quantum circuits by using the sampling algorithms coded
in src/main

"""

def get_HOGs():
    for num_qubits in range(8, 9):
        alg = get_sampling_algorithm(num_qubits)
        alg.writeHog()

def exp2():
    # Sampling using simulator.run() method from cirq
    qubits  = get_sycamore23_qubits(0)[15:]
    circuit = get_sycamore_circuit(qubits, 20)
    simulator = cirq.Simulator()
    print("hey")
    result = sample_circuit(circuit, qubits, 10, simulator)
    print(result)

def plot_XEBs(num_qubits=4):
    alg = get_sampling_algorithm(num_qubits)
    XEBs = alg.get_XEBs(num_qubits)
    orders = np.arange(num_qubits+1)
    plt.style.use('seaborn')
    plt.plot(orders, XEBs,  label='ideal')
    plt.scatter(orders, XEBs)
    plt.xlabel("Order of Correlators")
    plt.ylabel("XEB")
    plt.show()

def exp4(alg):
    N = 10
    tries = defaultdict(list)
    func_calls = [10, 100, 250, 500, 750, 1000, 1500]
    for num_events in func_calls:
        for _ in range(N):
            print(num_events)
            tries[num_events].append(2**alg.num_qubits * alg.get_exp_Hog(order=alg.num_qubits, events=num_events) - alg.correlators[0])
    return tries

def get_experimental_XEBs_with_sem():
    # plot ideal XEBs and mean of 5 experiments with error bars
    # 1. plot ideal XEB
    alg = get_sampling_algorithm(13, seed=42)
    ideal_xebs = alg.get_XEBs(alg.num_qubits)
    orders = np.arange(alg.num_qubits+1)
    # get experimental XEBs N times
    N = 5
    result = np.copy(orders)
    plt.cla()
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10,8))
    for simulation in range(N):
        # simulate a different circuit
        alg = get_sampling_algorithm(13, seed=simulation*69)
        hogs = []
        for order in orders:
            alg.marginals.clear()
            hogs.append(alg.get_experimental_Hog(order, events=int(1e3)))
        xebs = 2**alg.num_qubits * np.array(hogs) - alg.correlators[0]
        result = np.vstack((result, xebs))
        plt.scatter(orders, xebs)
    plt.plot(orders, ideal_xebs, label='Ideal XEB')
    avg_exp_xebs = np.mean(result[1:,:], axis=0)
    plt.plot(orders, avg_exp_xebs, label='Experimental XEB')
    plt.errorbar(orders, avg_exp_xebs, yerr=sem(result[1:,:], axis=0), fmt='x', capsize=3)
    plt.xlabel('Order of correlators')
    plt.ylabel('XEB')
    plt.suptitle('XEB vs order of correlators')
    plt.legend()
    plt.savefig('plotted_xeb')
    plt.show()

def timeit_benchmarks():
    results = []
    step=3
    for i in np.arange(3, 18+step, step):
        alg = get_sampling_algorithm(num_qubits=i, slow=False)
        # foo = %timeit -o -r 100 -n 7 alg.sample_events_test()
        # results.append(foo)
    # write results somewhere instead of returning
    return results
    

if __name__ == "__main__":
    # plot_XEBs()
    # kwargs = {'num_qubits': 3, 'seed': 14122000, 'slow': False}
    # alg = get_sampling_algorithm(**kwargs)
    # stmt = "timeit_benchmarks()"
    # step=3
    # for num_qubits in range(3, 21+step, step=step):
    # print(timeit.timeit("timeit_benchmarks(alg)", globals=globals(), number=100))
    # timeit_benchmarks()
    get_HOGs()

