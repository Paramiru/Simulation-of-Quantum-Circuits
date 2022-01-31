import numpy as np

class SamplingAlgorithm():
    def __init__(self, fourier_coeff): 
        self.N = np.log2(len(fourier_coeff))
        if self.N % 1 != 0:
            raise ValueError("fourier_coeff should be an array of length power of 2")
        self.fourier_coeff = fourier_coeff

    def __get_correlators(self, y, fc, VERBOSE):
        # y is a string containing a number in binary
        k = len(y)
        y = int(y, 2)
        correlators = np.array([])
        for i in range(2**k):
            bitstring = bin(i)[2:]
            # obtain fourier coefficient of that bitstring
            correlator = fc[i]
            # compute parity function Chi_s(y)
            sign = (-1) ** (bin(i & y).count('1') % 2)
            # add correlator to list
            correlators = np.append(correlators, sign * correlator)
            if VERBOSE:
                print(f"Bitstring: {bitstring}    \t Fourier coeff: {fc[i]:.5f}\ \t \
                    Correlator: {2**k * correlator * sign:.5f} \t sign of correlator: {sign}")
        # remember correlators have a 2**n term, with n being size of the bitstring
        return 2**k * correlators

    def __get_prob_add_zero(self, y, prob_limit, VERBOSE):
        k = len(y)
        correlators = self.__get_correlators(y, self.fourier_coeff, VERBOSE)
        return 0.5 * (prob_limit + sum(correlators)/(2**k))

    def sample_random_circuit(self, VERBOSE):
        # if N > np.log2(len(self.fourier_coeff)):
        #     raise ValueError("N cannot be bigger than the size of the circuit")
        outcome = ''
        prob_limit = 1 # p(0) + p(1) = 2p_hat(0)
        prob_add_zero = self.fourier_coeff[0]
        # N is length of bitstrings. Add check for N to be <= size of qubits in the circuit
        for _ in range(self.N.astype(int)):
            if np.random.default_rng().uniform(low=0, high=prob_limit) <= prob_add_zero:
                outcome += '0'
                prob_limit = prob_add_zero
            else:
                outcome += '1'
                prob_limit = 1 - prob_add_zero
            prob_add_zero = self.__get_prob_add_zero(outcome, prob_limit, VERBOSE)
        return outcome
    
    def sample_events(self, N: int, k: int, VERBOSE=False):
        # TODO: remove hardcoded values with global variables 
        if (k > 23):
            raise ValueError("k is {}, but it must be smaller or equal to {},\
            which corresponds to the number of qubits in the circuit")
        print()
        for i in range(1,N+1):
            sample = self.sample_random_circuit(VERBOSE)
            print(f"Outcome {i} is \'{sample}\'")