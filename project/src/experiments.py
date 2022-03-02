from cirq_random_test import get_sampling_algorithm

"""Examples of Quantum Simulations

This script allows the user to try several ways in which to sample
different quantum circuits by using the sampling algorithms coded
in project/main
"""

def exp1():
    alg = get_sampling_algorithm(13)
    alg.writeHog()
    alg = get_sampling_algorithm(16)
    alg.writeHog()

if __name__ == "__main__":
    exp1()