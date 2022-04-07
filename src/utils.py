import gmpy2
import numpy as np
import numpy.typing as npt


def square_mod(
    arr: npt.NDArray
) -> npt.NDArray:
    """Computes square modulus to a complex vector element-wise
    |a + bj| ^ 2 = (a + bj)(a - bj)
    
    Args
    ----
    arr : ndarray
        Array to which apply the square modulus

    Returns
    -------
    result : ndarray
        Resulting array after applying the square modulus
    """
    return np.abs(arr) ** 2

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

