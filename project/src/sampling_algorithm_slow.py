import numpy as np
from sampling_algorithm import SamplingAlgorithm

class SamplingAlgorithmSlow(SamplingAlgorithm):
    def __init__(self, correlators, order_arr, seed=14122000):
        super().__init__(correlators, order_arr, seed)

    def get_correlators_for_marginal_slow(self, y, VERBOSE):
        # Brute force method for the first implementation
        # TODO: Benchmark against get_correlators_for_marginal_to_order
        k = len(y)
        # y is a bitstring containing a number in binary
        y = int(y, 2)
        correlators = np.array([])
        for i in range(2**k):
            s_z = bin(i)[2:]
            index = s_z + '1'
            index = int(index + '0' * (self.num_qubits-len(index)), 2)
            # obtain correlator for that bitstring plus 1
            # index = int(s_z + '1', 2)
            correlator = self.correlators[index]
            # compute parity function Chi_s(y)
            sign = (-1) ** (bin(i & y).count('1') % 2)
            # add correlator to list
            correlators = np.append(correlators, sign * correlator)
        return correlators

    def get_prob_add_zero_slow(self, y, prob_limit):
        l = len(y)
        correlators = self.get_correlators_for_marginal_slow(y, False)
        return 0.5 * (prob_limit + np.sum(correlators)/(2**l))

    def sample_random_circuit_slow(self):
        outcome = ''
        prob_limit = 1 # p(0) + p(1) = 2p_hat(0)
        idx0, idx1 = 0, int('1'+'0'*(self.num_qubits-1), 2)
        prob_add_zero = 0.5 * (self.correlators[idx0] + self.correlators[idx1])
        # Looping to obtain value of each qubit sequentially
        # self.rng = np.random.default_rng(SeedSequence(14122000))
        for step in range(self.num_qubits):
            flipped_coin = self.rng.uniform(low=0, high=prob_limit)
            # print(getPrintProbString(step, outcome, prob_add_zero, prob_limit, flipped_coin))
            if flipped_coin <= prob_add_zero:
                outcome += '0'
                prob_limit = prob_add_zero
            else:
                outcome += '1'
                prob_limit = prob_limit - prob_add_zero
            if (step != self.num_qubits - 1):
                prob_add_zero = self.get_prob_add_zero_slow(outcome, prob_limit)
        return outcome

    # delete before submitting
    def sample_events_slow_test(self, num_outcomes=100):
        self.marginals.clear()
        for _ in range(num_outcomes):
            self.sample_random_circuit_slow()
