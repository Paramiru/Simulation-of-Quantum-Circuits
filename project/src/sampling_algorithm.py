from collections import defaultdict

import numpy as np
import pandas as pd
from numpy.random import SeedSequence

from circuitUtils import get_order_array
from IO import getPrintProbString


class SamplingAlgorithm():

    def __init__(self, correlators): 
        self.num_qubits = np.log2(len(correlators)).astype(int)
        if self.num_qubits % 1 != 0:
            raise ValueError("correlators should be an array of length power of 2")
        self.order_arr = get_order_array(self.num_qubits)
        self.correlators = np.array(correlators, dtype='float64')
        self.orders_and_correlators = np.array(list(zip(self.order_arr, self.correlators)))
        # if key is bitstring y, value is marginal prob of y + '0'
        self.marginals = defaultdict(np.float64)
        # Initialise random number generator
        self.rng = np.random.default_rng(SeedSequence(123))

    def get_marginal_p_0(self, order):
        self.marginals[''] = 1
        idx0, idx1 = 0, int('1'+'0'*(self.num_qubits-1), 2)
        if order == 0:
            prob_add_zero = 0.5 * self.correlators[idx0]
        else:
            prob_add_zero = 0.5 * (self.correlators[idx0] + self.correlators[idx1])
        self.marginals['0'] = prob_add_zero
        self.marginals['1'] = self.marginals[''] - prob_add_zero

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
    
    def get_index(self, y, x):
        s_z = bin(x)[2:]
        idx = (len(y) - len(s_z)) * '0' + s_z + '1'
        return int(idx + '0' * (self.num_qubits-len(idx)), 2)

    def get_correlators_for_marginal_to_order(self, y, order) -> np.ndarray:
        if order > self.num_qubits:
            raise ValueError("The order of the correlators cannot be bigger \
                than their maximum Hamming distance")
        k = len(y)
        idxs = np.arange(2**k)
        # obtain indexes for correlators for marginal p(y + '1')
        # TODO: check if this can be done
        new_idxs = np.array([self.get_index(y, x) for x in idxs])
        # use this mask to remove indexes that consider correlators
        # of higher order. Masking avoids looping over all the array
        mask = self.orders_and_correlators[new_idxs][:,0] <= order
        corr_idxs = new_idxs[mask]
        # turn bitstring to integer
        int_y = int(y, 2)
        # compute exponent of parities. Use idxs and not corr_idxs which
        # have a '1' appended at the end
        parity_exp_products = (idxs[mask] & int_y)[:,None]
        # np.unpackbits can only be used with 'uint8'
        parity_exp_products = parity_exp_products.view('uint8')
        # Use unpackbits to avoid looping and get the hamming distance of
        # s_z * y in the equation
        bin_array = np.unpackbits(
            parity_exp_products,
            axis=1,
            count=self.num_qubits,
            bitorder='little'
        )
        parity_exp = np.sum(bin_array, axis=1) % 2
        # Turn 0 and 1 to +1 and -1 to add the sign for the correlators
        parity_exp = ~parity_exp.astype(bool)
        parity = np.left_shift(parity_exp.astype('int8'), 1) - 1
        return parity * self.correlators[corr_idxs]

    def get_prob_add_zero(self, y, prob_limit, order):
        l = len(y)
        correlators = self.get_correlators_for_marginal_to_order(y, order)
        return 0.5 * (prob_limit + np.sum(correlators)/(2**l))

    def get_prob_add_zero_slow(self, y, prob_limit):
        l = len(y)
        correlators = self.get_correlators_for_marginal_slow(y, False)
        return 0.5 * (prob_limit + np.sum(correlators)/(2**l))

    def sample_random_circuit(self, order, VERBOSE):
        outcome = ''
        prob_limit = 1 # p(0) + p(1) = 2p_hat(0)
        self.get_marginal_p_0(order)
        # self.rng = np.random.default_rng(SeedSequence(14122000))
        # Looping to obtain value of each qubit sequentially
        for step in range(self.num_qubits):
            prob_add_zero = self.marginals[outcome + '0']
            flipped_coin = self.rng.uniform(low=0, high=prob_limit)
            if VERBOSE:
                print(getPrintProbString(step, outcome, prob_add_zero, prob_limit, flipped_coin))
            if flipped_coin <= prob_add_zero:
                outcome += '0'
                prob_limit = prob_add_zero
            else:
                outcome += '1'
                prob_limit = prob_limit - prob_add_zero
            if (step != self.num_qubits - 1):
                if not self.marginals[outcome + '0']:
                    self.marginals[outcome + '0'] = self.get_prob_add_zero(outcome, prob_limit, order)
                    # do not need to store prob of adding one. We only use prob of adding 0 
                    # to obtain what each qubit has to be
        if outcome[-1] == '1':
            self.marginals[outcome] = prob_limit 
        return outcome

    def sample_random_circuit_slow(self):
        outcome = ''
        prob_limit = 1 # p(0) + p(1) = 2p_hat(0)
        idx0, idx1 = 0, int('1'+'0'*(self.num_qubits-1), 2)
        prob_add_zero = 0.5 * (self.correlators[idx0] + self.correlators[idx1])
        # Looping to obtain value of each qubit sequentially
        self.rng = np.random.default_rng(SeedSequence(14122000))
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
    
    def sample_events(self, num_outcomes: int, order: int, VERBOSE):
        if (order > self.num_qubits):
            raise ValueError("The order must be smaller or equal to " + self.num_qubits)
        prob_of_samples = np.array([])
        for i in range(num_outcomes):
            sample = self.sample_random_circuit(order, VERBOSE)
            prob_of_samples = np.append(prob_of_samples, self.marginals[sample])
            print(f"Outcome {i} is |{sample}>")
        return prob_of_samples

    def add_marginal(self, outcome: str, pruning_depth: int, order: int, prob_limit: np.float64):
        if len(outcome) == pruning_depth: return
        marginal = self.get_prob_add_zero(
            outcome,
            prob_limit,
            order
        )
        self.marginals[outcome + '0'] = marginal
        self.marginals[outcome + '1'] = prob_limit - marginal
        self.add_marginal(outcome + '0', pruning_depth, order, marginal)
        self.add_marginal(outcome + '1', pruning_depth, order, self.marginals[outcome + '1'])

    def learning_marginals(self, pruning_depth: int, order: int):
        if pruning_depth < 0 or pruning_depth > self.num_qubits:
            raise ValueError("pruning_depth should be between 0 and the number of qubits")
        self.get_marginal_p_0(order)
        self.add_marginal('0', pruning_depth, order, self.marginals['0'])
        self.add_marginal('1', pruning_depth, order, self.marginals['1'])         

    def writeHog(self):
        orders = np.arange(self.num_qubits+1)
        data = dict()
        data['order'] = orders
        HOGs = []
        for order in orders:
            self.marginals.clear()
            HOGs.append(np.mean(self.sample_events(10000, order, False)))
        data['HOGs'] = HOGs
        df = pd.DataFrame.from_dict(data)
        df.to_csv('order-and-HOG-' + str(self.num_qubits))
