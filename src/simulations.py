import time
from typing import Iterable

import cirq
from cirq.sim.state_vector_simulator import StateVectorTrialResult

from circuitUtils import (get_simple_circuit, get_sycamore23_qubits,
                          get_sycamore_circuit)


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
