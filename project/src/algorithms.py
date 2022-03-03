import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose

from sampling_algorithm import SamplingAlgorithm
from simulations import simulate_sycamore_circuit
from utils import get_order_array, square_mod


def fwht(a: npt.NDArray):
    """In-place Fast Walsh–Hadamard Transform of array a
    
    Args
    ----
    a : npt.NDArray
        Array to Walsh-Hadamard Transform in place
    """

    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2

def get_fourier_cf(qn_state: npt.NDArray) -> npt.NDArray:
    """Obtain Fourier coefficients from a given vector of length
    2^n. Uses the Fast Welsch-Hadamard Transform algorithm

    Args
    ----
    qn_state : npt.NDArray
        Quantum state vector from which to obtain the Fourier
        coefficients
    
    Returns
    -------
    fourier_cf : npt.NDArray
        Array containing the Fourier coefficients from the input
        quantum state qn_state
    """

    fourier_cf = np.ndarray.copy(qn_state)
    fwht(fourier_cf)
    return fourier_cf


def get_sampling_algorithm(num_qubits: int) -> SamplingAlgorithm:
    # num_qubits = int(input("Number of qubits between 1 and 23: "))
    result = simulate_sycamore_circuit(N=num_qubits)
    # result, N = simulate_basic_circuit(), 2

    # Print the final state vector (wavefunction).
    q_state = result.final_state_vector
    print(f"\nState vector:\n{q_state}")

    # Obtain probabilities for each state
    q_state = square_mod(q_state)
    print(f"\nProbability vector:\n{q_state}")

    # Applying Welsch-Hadamard transform to obtain Fourier coefficients
    correlators = get_fourier_cf(q_state)
    print(f"\nArray of correlators:\n{correlators}")

    # plot_XEB_for_every_k(num_qubits, correlators)
    return SamplingAlgorithm(correlators, get_order_array(num_qubits))


def test_fwht():
    a = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    expected_output_a = [4, 2, 0, -2, 0, 2, 0, 2]
    fwht(a)
    assert_allclose(a, expected_output_a)

    b = np.array([4, 2, 2, 0, 0, 2, -2, 0])
    expected_output_b = [8, 0, 8, 0, 8, 8, 0, 0]
    fwht(b)
    assert_allclose(b, expected_output_b)

    c = np.array([1, 0])
    fwht(c)
    print(c)

    d = np.array([0, 1])
    fwht(d)
    print(d)

if __name__ == "__main__":
    test_fwht()
