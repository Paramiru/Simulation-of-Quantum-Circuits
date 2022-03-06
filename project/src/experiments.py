import cirq
import numpy as np
import matplotlib.pyplot as plt

from algorithms import get_sampling_algorithm
from circuitUtils import (get_sycamore23_qubits, get_sycamore_circuit,
                          sample_circuit)


"""Examples of Quantum Simulations

This script allows the user to try several ways in which to sample
different quantum circuits by using the sampling algorithms coded
in project/main
"""

def exp1():
    for num_qubits in range(1, 24):
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

def exp3():
    for num_qubits in range(5, 6):
        alg = get_sampling_algorithm(num_qubits)
        XEBs = alg.get_XEBs(num_qubits)
        orders = np.arange(num_qubits)
        plt.style.use('seaborn-whitegrid')
        plt.plot(orders, XEBs,  label='ideal')
        plt.scatter(orders, XEBs)
        plt.xlabel("Order of Correlators")
        plt.ylabel("XEB")

if __name__ == "__main__":
    exp1()
    