import numpy as np

class SamplingAlgorithm():
    def __init__(self, correlators): 
        self.N = np.log2(len(correlators)).astype(int)
        if self.N % 1 != 0:
            raise ValueError("correlators should be an array of length power of 2")
        self.correlators = correlators

    def __get_correlators_for_marginal(self, y, VERBOSE):
        k = len(y)
        # y is a bitstring containing a number in binary
        y = int(y, 2)
        correlators = np.array([])
        for i in range(2**k):
            s_z = bin(i)[2:]
            # obtain correlator for that bitstring plus 1
            index = int(s_z + '1', 2)
            correlator = self.correlators[index]
            # compute parity function Chi_s(y)
            sign = (-1) ** (bin(i & y).count('1') % 2)
            # add correlator to list
            correlators = np.append(correlators, sign * correlator)
            if VERBOSE:
                print(f"Bitstring: {s_z} + '1'    \t Correlator: {self.correlators[s_z + '1']:.5f}\ \t \
                    Correlator: {2**k * correlator * sign:.5f} \t sign of correlator: {sign}")
        return correlators

    def __get_prob_add_zero(self, y, prob_limit, VERBOSE):
        l = len(y) + 1
        correlators = self.__get_correlators_for_marginal(y, VERBOSE)
        return 0.5 * (prob_limit + sum(correlators)/(2**l))

    def sample_random_circuit(self, k:int, VERBOSE=False):
        outcome = ''
        prob_limit = 1 # p(0) + p(1) = 2p_hat(0)
        idx0, idx1 = 0, int('1'+'0'*(self.N-1), 2)
        prob_add_zero = 0.5 * (self.correlators[idx0] + self.correlators[idx1])
        # print(f"Prob_add_zero: {prob_add_zero}")
        # Initialise random number generator
        rng = np.random.default_rng()
        for step in range(k):
            # print(f"Step {step}, outcome so far is |{outcome}>")
            # print(f"\nStep {step}")
            # print("-------------")
            # print(f"Current outcome is |{outcome}>")
            # print(f"Probability of adding 0 is {prob_add_zero:.3f}")
            # print(f"Probability of adding 1 is {(prob_limit-prob_add_zero):.3f}")
            # print(f"p({outcome}0) + p({outcome}1) = {prob_limit:.3f}")
            flipped_coin = rng.uniform(low=0, high=prob_limit)
            # print(f"Random number generated is {flipped_coin:.3f}")
            if flipped_coin <= prob_add_zero:
                outcome += '0'
                prob_limit = prob_add_zero
            else:
                outcome += '1'
                prob_limit = prob_limit - prob_add_zero
            
            if (step != k - 1):
                prob_add_zero = self.__get_prob_add_zero(outcome, prob_limit, VERBOSE)
        return outcome
    
    def sample_events(self, num_outcomes: int, num_qubits: int, VERBOSE=False):
        # TODO: remove hardcoded values with global variables 
        if (num_qubits > self.N):
            raise ValueError(f"k is {num_qubits}, but it must be smaller or equal to {self.N},\
            which corresponds to the number of qubits in the simulated circuit")
        for i in range(1, num_outcomes+1):
            sample = self.sample_random_circuit(num_qubits, VERBOSE)
            print(f"\nOutcome {i} is |{sample}>")