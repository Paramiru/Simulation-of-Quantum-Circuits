from collections import defaultdict

import numpy as np
import pandas as pd
from numpy.random import SeedSequence

from circuitUtils import get_index
from IO import getPrintProbString
from utils import square_mod


class SamplingAlgorithm():
    def __init__(self, correlators, order_arr, seed=14122000): 
        self.num_qubits = np.log2(len(correlators)).astype(int)
        if self.num_qubits % 1 != 0:
            raise ValueError("correlators should be an array of length power of 2")
        # Get order of each Fourier coefficient / correlator
        self.order_arr = order_arr
        self.correlators = np.array(correlators, dtype='float64')
        self.orders_and_correlators = np.array(list(zip(self.order_arr, self.correlators)))
        # if key is bitstring y, value is marginal prob of y + '0'
        self.marginals = defaultdict(np.float64)
        # Initialise random number generator
        self.rng = np.random.default_rng(SeedSequence(seed))

    def get_marginal_p_0(self, order):
        self.marginals[''] = 1
        idx0, idx1 = 0, int('1'+'0'*(self.num_qubits-1), 2)
        if order == 0:
            prob_add_zero = 0.5 * self.correlators[idx0]
        else:
            prob_add_zero = 0.5 * (self.correlators[idx0] + self.correlators[idx1])
        self.marginals['0'] = prob_add_zero
        self.marginals['1'] = self.marginals[''] - prob_add_zero

    def get_prob_add_zero(self, y, prob_limit, order):
        l = len(y)
        correlators = self.get_correlators_for_marginal_to_order(y, order)
        return 0.5 * (prob_limit + np.sum(correlators)/(2**l))

    def get_correlators_for_marginal_to_order(self, y, order) -> np.ndarray:
        if order > self.num_qubits:
            raise ValueError("The order of the correlators cannot be bigger \
                than their maximum Hamming distance")
        k = len(y)
        idxs = np.arange(2**k)
        # obtain indexes for correlators for marginal p(y + '1')
        new_idxs = np.array([get_index(y, x, self.num_qubits) for x in idxs])
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

    def sample_random_circuit(self, order, VERBOSE=False):
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
                    # Change marginal of  'outcome' + '0' if it's > marginal of 'outcome'
                    # to avoid negative probabilities
                    if self.marginals[outcome + '0'] > prob_limit:
                        self.marginals[outcome + '0'] = prob_limit
        if outcome[-1] == '1':
            self.marginals[outcome] = prob_limit 
        return outcome

    # delete before submitting
    def sample_random_circuit_test(self):
        outcome = ''
        prob_limit = 1 # p(0) + p(1) = 2p_hat(0)
        self.get_marginal_p_0(self.num_qubits)
        # self.rng = np.random.default_rng(SeedSequence(14122000))
        # Looping to obtain value of each qubit sequentially
        for step in range(self.num_qubits):
            prob_add_zero = self.marginals[outcome + '0']
            flipped_coin = self.rng.uniform(low=0, high=prob_limit)
            if flipped_coin <= prob_add_zero:
                outcome += '0'
                prob_limit = prob_add_zero
            else:
                outcome += '1'
                prob_limit = prob_limit - prob_add_zero
            if (step != self.num_qubits - 1):
                if not self.marginals[outcome + '0']:
                    self.marginals[outcome + '0'] = self.get_prob_add_zero(outcome, prob_limit, self.num_qubits)
                    # do not need to store prob of adding one. We only use prob of adding 0 
                    # Change marginal of  'outcome' + '0' if it's > marginal of 'outcome'
                    # to avoid negative probabilities
                    if self.marginals[outcome + '0'] > prob_limit:
                        self.marginals[outcome + '0'] = prob_limit
        if outcome[-1] == '1':
            self.marginals[outcome] = prob_limit 
        return outcome

    # delete before submitting
    def sample_events_test(self, num_outcomes=100):
        # clear marginals to do every test independent of each other
        self.marginals.clear()
        for _ in range(num_outcomes):
            self.sample_random_circuit_test()
    
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
        print(f"Outcome: {outcome}, marginal {marginal}, prob_limit: {prob_limit}")
        self.marginals[outcome + '0'] = marginal if marginal <= prob_limit else prob_limit 
        print(f"Marginal of {outcome + '0'}: {self.marginals[outcome + '0']}")
        self.marginals[outcome + '1'] = prob_limit - self.marginals[outcome + '0']
        print(f"Marginal of {outcome + '1'}: {self.marginals[outcome + '1']}")
        self.add_marginal(outcome + '0', pruning_depth, order, marginal)
        self.add_marginal(outcome + '1', pruning_depth, order, self.marginals[outcome + '1'])

    def learning_marginals(self, pruning_depth: int, order: int):
        if pruning_depth < 0 or pruning_depth > self.num_qubits:
            raise ValueError("pruning_depth should be between 0 and the number of qubits")
        self.get_marginal_p_0(order)
        self.add_marginal('0', pruning_depth, order, self.marginals['0'])
        self.add_marginal('1', pruning_depth, order, self.marginals['1']) 

    def get_XEB(self, k: int) -> np.float64:
        mask = self.order_arr <= k
        mask[0] = False
        correlators_upto_order_k = mask * self.correlators
        return np.sum(square_mod(correlators_upto_order_k))

    def get_XEBs(self, N: int) -> np.ndarray:
        orders = np.arange(N + 1)
        return np.array([self.get_XEB(order) for order in orders])  

    def get_experimental_Hog(self, order, events=int(1e4)):
        return np.mean(self.sample_events(events, order, False))

    def writeHog(self):
        orders = np.arange(self.num_qubits+1)
        data_ideal = dict()
        data_exp = dict()
        data_ideal['order'] = orders
        data_ideal['XEB'] = self.get_XEBs(self.num_qubits)
        data_exp['order'] = orders
        HOGs = []
        for order in orders:
            print(f"Doing order {order}")
            self.marginals.clear()
            HOGs.append(self.get_experimental_Hog(order))
        data_exp['HOGs'] = HOGs
        df_exp = pd.DataFrame.from_dict(data_exp)
        df_exp.to_scv('LongjobWorked.csv')
        # df_exp.to_csv('results/new-order-HOG-experimental' + str(self.num_qubits))
        df_ideal = pd.DataFrame.from_dict(data_ideal)
        # df_ideal.to_csv('results/new-order-HOG-ideal' + str(self.num_qubits))
        df_ideal.to_csv('LongjobWorkedYeah.csv')
