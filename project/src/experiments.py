from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import seaborn as sns

from algorithms import get_sampling_algorithm


"""Examples of Quantum Simulations

This script allows the user to try several ways in which to sample
different quantum circuits by using the sampling algorithms coded
in src/main

"""

def get_HOGs():
    for num_qubits in range(1, 15):
        alg = get_sampling_algorithm(num_qubits)
        alg.writeHog()

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

def get_error_on_XEB_with_num_of_samples(alg):
    N = 10
    tries = defaultdict(list)
    func_calls = [10, 100, 250, 500, 750, 1000, 1500]
    for num_events in func_calls:
        for _ in range(N):
            print(num_events)
            tries[num_events].append(2**alg.num_qubits *
                alg.get_experimental_Hog(order=alg.num_qubits, events=num_events)
                - alg.correlators[0])
    return tries

def plot_error_on_XEB_with_num_of_samples(num_qubits, seed=42):
    alg = get_sampling_algorithm(num_qubits, seed)
    tries = get_error_on_XEB_with_num_of_samples(alg)
    N = len(tries[100])
    func_calls = np.array([10, 100, 250, 500, 750, 1000, 1500])[:,None]
    xs = np.repeat(func_calls, repeats=N, axis=1)
    plt.rcParams.update({'font.size': 20})
    sns.set_style('ticks')

    plt.figure(figsize=(15,9))
    plt.errorbar(10, np.mean(tries[10]), yerr=np.std(tries[10]), fmt='x', capsize=5)
    plt.scatter(xs[0], tries[10])

    plt.errorbar(100, np.mean(tries[100]), yerr=np.std(tries[100]), fmt='x',
                capsize=5)
    plt.scatter(xs[1], tries[100])

    plt.scatter(xs[2], tries[250])
    plt.errorbar(250, np.mean(tries[250]), yerr=np.std(tries[250]), fmt='x',
                capsize=5)

    plt.scatter(xs[3], tries[500])
    plt.errorbar(500, np.mean(tries[500]), yerr=np.std(tries[500]), fmt='x',
                capsize=5)

    plt.scatter(xs[-3], tries[750])
    plt.errorbar(750, np.mean(tries[750]), yerr=np.std(tries[750]), fmt='x',
                capsize=5)

    plt.scatter(xs[-2], tries[1000])
    plt.errorbar(1000, np.mean(tries[1000]), yerr=np.std(tries[1000]), fmt='x',
                capsize=5)

    plt.scatter(xs[-1], tries[1500])
    plt.errorbar(1500, np.mean(tries[1500]), yerr=np.std(tries[1500]), fmt='x',
                capsize=5)

    ideal_XEB = alg.get_XEB(alg.num_qubits)
    plt.axhline(y = ideal_XEB, color = 'r', linestyle = '--', label='ideal XEB')

    plt.xlabel('Samples per experiment', fontsize=20)
    plt.ylabel('XEB', fontsize=20)

    plt.xticks([10, 100, 250, 500, 750, 1000, 1500], fontsize=20)
    plt.legend(loc='best', prop={'size': 14})
    plt.suptitle('Experimental XEB vs number of samples generated', fontsize=22)
    # plt.savefig('XEB-vs-number-of-samples.pdf')
    plt.show()

def plot_experimental_XEBs_with_sem(num_qubits=5, seed=42):
    # plot ideal XEBs and mean of 5 experiments with error bars
    # 1. plot ideal XEB
    alg = get_sampling_algorithm(num_qubits, seed=seed)
    ideal_xebs = alg.get_XEBs(alg.num_qubits)
    orders = np.arange(alg.num_qubits+1)
    # get experimental XEBs N times
    N = 5
    result = np.copy(orders)
    plt.cla()
    plt.figure(figsize=(15,8))
    # sns.set_theme()
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 20})
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for simulation in range(N):
        # simulate a different circuit
        alg = get_sampling_algorithm(num_qubits, seed=simulation*69)
        hogs = []
        for order in orders:
            alg.marginals.clear()
            hogs.append(alg.get_experimental_Hog(order, events=int(1e3)))
        xebs = 2**alg.num_qubits * np.array(hogs) - alg.correlators[0]
        result = np.vstack((result, xebs))
        plt.scatter(orders, xebs, c=colors[simulation], label=f'Experiment {simulation+1}')
    plt.plot(orders, ideal_xebs, label='Ideal XEB')
    avg_exp_xebs = np.mean(result[1:,:], axis=0)
    plt.plot(orders, avg_exp_xebs, label='Mean of experimental XEB')
    plt.errorbar(orders, avg_exp_xebs, yerr=sem(result[1:,:], axis=0), fmt='x',
            ecolor = 'mediumslateblue', markersize='10', color='m', capsize=5)
    plt.xlabel('Order of correlators')
    plt.xticks(orders)
    plt.ylabel('XEB')
    plt.suptitle('XEB vs order of correlators', fontsize=24)
    plt.legend(prop={'size': 16})
    filename = f'plotted_xeb_seed_{seed}.pdf'
    plt.savefig(filename)
    plt.show()

def on_the_fly_sampling(num_qubits, events, VERBOSE):
    alg = get_sampling_algorithm(num_qubits,  slow=True, VERBOSE=VERBOSE)
    alg.sample_events(events)

def hybrid_sampling(num_qubits, events, order, VERBOSE=True):
    alg = get_sampling_algorithm(num_qubits, slow=False, VERBOSE=VERBOSE)
    # Uncomment and modify to have partial learning phase:
    # alg.learning_marginals(pruninc_depth=alg.num_qubits//2, order=alg.num_qubits)
    alg.sample_events(events, order, alg.sample_random_circuit_hybrid)

def fast_sampling(num_qubits, events, order, VERBOSE):
    alg = get_sampling_algorithm(num_qubits, slow=False, VERBOSE=VERBOSE)
    print(f"\nStarting learning phase.")
    alg.learning_marginals(pruning_depth=alg.num_qubits, order=alg.num_qubits)
    print(f"\nLearning phase successfully completed.\n")
    alg.sample_events(events, order, alg.sample_random_circuit_fast)



if __name__ == "__main__":
    # Uncomment lines to try different experiments
    params = {'num_qubits': 12, 'events': 1000, 'VERBOSE': 1} 

    # on_the_fly_sampling(**params)

    # hybrid_sampling(**params, order=10)

    # fast_sampling(**params, order=12)

    # use num_qubits=8 minimum to get a decent plot.
    plot_experimental_XEBs_with_sem(10)

    # plot_error_on_XEB_with_num_of_samples(num_qubits=10)
