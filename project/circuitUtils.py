import cirq
import cirq_google

from typing import List, Tuple
from collections.abc import Iterable

import time

import numpy as np

import gmpy2


def get_extra_qubits_for_sycamore() -> List[cirq.GridQubit]:
    """Returns a list of 12 qubits. These correspond to the next
    12 qubits that Google's Sycamore processor would have after 
    qubit number 23

    Returns
    -------
    qubits : Iterable[cirq.GridQubits]
        a list of 12 GridQubits to use for extending the qubits
        from Sycamore23
    """

    q0 = cirq.GridQubit(3,3)
    q1 = cirq.GridQubit(4,4)
    q2 = cirq.GridQubit(5,5)
    q3 = cirq.GridQubit(6,6)
    q4 = cirq.GridQubit(2,3)
    q5 = cirq.GridQubit(3,4)
    q6 = cirq.GridQubit(4,5)
    q7 = cirq.GridQubit(5,6)
    q8 = cirq.GridQubit(6,7)
    q9 = cirq.GridQubit(2,4)
    q10 = cirq.GridQubit(3,5)
    q11 = cirq.GridQubit(4,6)
    q12 = cirq.GridQubit(5,7)

    return [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12]

def get_sycamore_circuit(
    qubits: Iterable[cirq.GridQubit],
    depth: int = 10
) -> cirq.Circuit:
    """Generate a random quantum circuit based on the circuits used
    in the paper https://www.nature.com/articles/s41586-019-1666-5

    Args
    ----
    qubits : Iterable[cirq.GridQubit]
        The qubits to use in the circuit
    depth : int
        Depth of the circuit

    Returns
    -------
    circuit : cirq.Circuit
        Generated circuit with the given qubits and depth
    """

    return cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
        qubits = qubits,
        depth = depth,
        two_qubit_op_factory = (lambda a, b, _: cirq.ops.ISwapPowGate()(a, b)),
        pattern = cirq.experiments.GRID_STAGGERED_PATTERN,
        single_qubit_gates = (
            cirq.ops.X ** 0.5, cirq.ops.Y ** 0.5, 
            cirq.ops.PhasedXPowGate(phase_exponent=0.25, exponent=0.5)
        ),
        add_final_single_qubit_layer = True,
        seed = None
    )

def get_sycamore23_qubits(num_extra_qubits: int = 0) -> Iterable[cirq.GridQubit]:
    """Returns the qubits used to simulate a circuit. It is based on
    Google's Sycamore23 quantum chip and the user can add more qubits
    by uncommenting the lines below

    Args
    ----
    num_extra_qubits : int
        Number of qubits to add to the original 23 qubits of Sycamore23. 
        It is constrained to be between 0 and 12

    Returns
    -------
    qubits : Iterable[cirq.GridQubit]
        Sycamore23 circuits plus the extra qubits added
    """

    # See the value of extra_qubits is not larger than 12
    num_extra_qubits = 12 if num_extra_qubits > 12 else num_extra_qubits

    qubits = cirq_google.Sycamore23.qubits
    extra_qubits = get_extra_qubits_for_sycamore()[:num_extra_qubits]
    extra_qubits = extra_qubits[:num_extra_qubits]
    qubits.extend(extra_qubits)
    return qubits

def get_simple_circuit():
    """"Generate a simple 2-qubit circuit consisting of sqrt_x
    and CZ gates
    
    Args
    ----
    meas : bool
        Indicate whether to measure the qubits at the end of the 
        circuit

    Returns
    -------
    circuit : cirq.Circuit
        2D circuit of three layers with sqrt_x and CZ gates
    qubits : Iterable[cirq.GridQubit]
        list of the two qubits used in the circuit, in order
    """
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(1, 0)

    def basic_circuit(meas=True) -> Tuple[cirq.Circuit, Iterable[cirq.GridQubit]]:
        sqrt_x = cirq.X**0.5
        yield sqrt_x(q0), sqrt_x(q1)
        yield cirq.CZ(q0, q1)
        yield sqrt_x(q0), sqrt_x(q1)
        if meas:
            yield cirq.measure(q0, key='q0'), cirq.measure(q1, key='q1')

    circuit = cirq.Circuit()
    circuit.append(basic_circuit())
    
    return circuit, [q0, q1]

def sample_circuit(
    circuit: cirq.Circuit,
    qubits: Iterable[cirq.GridQubit],
    repetitions: int,
    simulator: cirq.Simulator
) -> None:
    """Sample the given circuit
    
    Args
    ----
    circuit : cirq.Circuit
        Circuit to sample from
    qubits : Iterable[cirq.GridQubit]
        Qubits to use
    repetitions : int
        Number of events to generate
    simulator : cirq.Simulator
        Simulator to use for the sampling

    Returns
    -------
    result : cirq.Result
        samples obtained after running the simulation
    """

    circuit.append(cirq.measure(*qubits, key='result'))
    tic = time.perf_counter()
    result = simulator.run(circuit, repetitions=repetitions)
    toc = time.perf_counter()
    print(f"Circuit sampled with {repetitions} repetitions in {toc - tic:0.4f} seconds")
    return result

def get_order_array(N: int = 23) -> np.array:
    """Obtain the corresponding order for each Fourier coefficient as an ordered array.
    Order is defined as the hamming distance of the computational basis state vector,
    which can be thought of as the number of 1s it contains.
    E.g. first element of the array corresponds to comp. basis state |0>^n hence order 0, 
    second element is 1 because the hamming distance of |0...1> is 1, etc.

    Args
    ----
    N : int
        Number of qubits the quantum circuit has
    
    Returns
    -------
    arr : np.array
        Array containing the order of the coefficients for each comp. basis state.
    
    """
    
    # tic    = time.perf_counter()
    seq = map(gmpy2.popcount, range(2**N))
    # toc    = time.perf_counter()
    # print(f"{toc - tic} seconds")
    return np.fromiter(seq, dtype=np.int32)
    

if __name__ == "__main__":
    r = get_order_array(23)
    print(r)