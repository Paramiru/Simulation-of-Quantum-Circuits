import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose
from onTheFlySampler import OnTheFlySampler

from Sampler import Sampler
from hybridSampler import HybridSampler
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

def get_correlators(qn_state: npt.NDArray) -> npt.NDArray:
    """Obtain Correlators (scaled Fourier coefficients) from a 
    given vector of length 2^n. Uses the Fast Welsch-Hadamard 
    Transform algorithm

    Args
    ----
    qn_state : npt.NDArray
        Quantum state vector from which to obtain the Fourier
        coefficients
    
    Returns
    -------
    correlators : npt.NDArray
        Array containing the Correlators from the input
        quantum state qn_state
    """

    correlators = np.ndarray.copy(qn_state)
    fwht(correlators)
    return correlators


def get_sampling_algorithm(
    num_qubits: int,
    seed=14122000,
    slow=False, 
    VERBOSE=False
    ) -> Sampler:
    
    result = simulate_sycamore_circuit(N=num_qubits)
    # Uncomment to try a simpler circuit
    # result, N = simulate_basic_circuit(), 2

    # Obtain the final state vector (wavefunction).
    q_state = result.final_state_vector
    if VERBOSE: print(f"\nState vector:\n{q_state}")

    # Obtain probabilities for each state
    q_state = square_mod(q_state)
    if VERBOSE: print(f"\nProbability vector:\n{q_state}")

    # Applying Welsch-Hadamard transform to obtain correlators
    correlators = get_correlators(q_state)
    if VERBOSE: print(f"\nArray of correlators:\n{correlators}")

    params = {'correlators': correlators, 'order_arr': get_order_array(num_qubits), 'seed': seed}
    return OnTheFlySampler(**params) if slow else HybridSampler(**params)


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
