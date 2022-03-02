import cirq
from cirq.sim.state_vector_simulator import StateVectorTrialResult

from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
import time

from sympy import fwht

from circuitUtils import get_sycamore23_qubits, get_sycamore_circuit, get_simple_circuit, get_order_array
from algorithms import square_mod, get_fourier_cf
from sampling_algorithm import SamplingAlgorithm

def simulate_circuit(
    circuit: cirq.Circuit,
    qubits: Iterable[cirq.GridQubit]
) -> StateVectorTrialResult:
    """"Simulate a circuit with the input qubits
    
    Args
    ----
    circuit : Circuit
        Circuit to simulate
    qubits : Iterable[GridQubit]
    
    Returns
    -------
    result : StateVectorTrialResult
        Final quantum state vector from the simulated circuit
    """

    simulator = cirq.Simulator()
    print(f'Simulating circuit with {len(qubits)} qubits')
    tic    = time.perf_counter()
    result = simulator.simulate(circuit, qubit_order=qubits)
    toc    = time.perf_counter()
    print(f"Circuit simulated in {toc - tic:0.4f} seconds")
    return result

def simulate_sycamore_circuit(
    N: int,
    depth: int = 20,
    num_extra_qubits: int = 0
) -> StateVectorTrialResult:
    """Simulate a quantum random circuit based in the ones used by
    Google in their 2019 quantum advantage paper
    
    Args
    ----
    depth : int
        The depth of the random circuit to be simulated
    num_extra_qubits : int
        Number of qubits to add apart from the ones of Sycamore23
    
    Returns
    -------
    result : StateVectorTrialResult
        Final quantum state vector from the simulated circuit    
    """
    if (N > 23 + num_extra_qubits):
        raise ValueError(f"N is larger than 23 + {num_extra_qubits}")
    qubits  = get_sycamore23_qubits(num_extra_qubits)
    circuit = get_sycamore_circuit(qubits[:N], depth)
    result  = simulate_circuit(circuit, qubits[:N])
    return result

def simulate_basic_circuit() -> StateVectorTrialResult:
    """Simulate a simple 2-qubit quantum circuit with sqrt(X) and
    CZ gates
    
    Returns
    -------
    result : StateVectorTrialResult
        Final quantum state vector from the simulated circuit    
    """

    basic_circuit, qubit_order = get_simple_circuit()
    result                     = simulate_circuit(basic_circuit, qubit_order)
    return result

def get_XEB(k, correlators):
    mask = order_arr <= k
    correlators_upto_order_k = mask * correlators
    return np.sum(square_mod(correlators_upto_order_k))

# def plot_HOB_for_every_k(N: int, fourier_coeff):
#     orders = np.arange(N+1)
#     HOBs = []
#     for i in orders:
#         HOBs.append(get_HOG(i, fourier_coeff))

#     # plt.scatter(orders, HOBs)
#     # plt.plot(orders, HOBs)
#     # plt.show()

def plot_XEB_for_every_k(N: int, correlators):
    orders = np.arange(N + 1)
    XEBs = []
    for i in orders:
        XEBs.append(get_XEB(i, correlators))
    XEBs = np.array(XEBs) - 1
    plt.style.use('seaborn-whitegrid')
    plt.plot(orders, XEBs)
    plt.scatter(orders, XEBs)
    plt.xlabel("Order of Correlators")
    plt.ylabel("XEB")
    plt.show()

def main():
    num_qubits = int(input("Number of qubits between 1 and 23: "))
    # Get order of each Fourier coefficient to use later
    global order_arr 
    order_arr = get_order_array(N=num_qubits)

    # np.random.seed(2)
    result = simulate_sycamore_circuit(N=num_qubits)
    # result, N = simulate_basic_circuit(), 2

    # Print the final state vector (wavefunction).
    q_state = result.final_state_vector
    print(f"\nState vector:\n{q_state}")

    # Obtain probabilities for each state
    q_state = square_mod(q_state)
    print(f"\nProbability vector:\n{q_state}")

    # Applying Welsch-Hadamard transform to obtain Fourier coefficients
    # correlators = fwht(q_state)
    correlators = get_fourier_cf(q_state)
    print(f"\nArray of correlators:\n{correlators}")

    plot_XEB_for_every_k(num_qubits, correlators)
    
    alg = SamplingAlgorithm(correlators)
    return alg
    # alg.writeHog()

if __name__ == "__main__":
    main()