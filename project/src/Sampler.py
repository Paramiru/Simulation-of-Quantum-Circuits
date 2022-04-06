from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy.random import SeedSequence

from circuitUtils import get_index
from utils import square_mod


class Sampler(ABC):
    def __init__(self, correlators, order_arr, seed=14122000): 
        self.num_qubits = np.log2(len(correlators)).astype(int)
        # Get order of each Fourier coefficient / correlator
        self.order_arr = order_arr
        self.correlators = np.array(correlators, dtype='float64')
        self.orders_and_correlators = np.array(list(zip(self.order_arr, self.correlators)))
        print(self.orders_and_correlators)
        # if key is bitstring y, value is marginal prob of y + '0'
        self.marginals = defaultdict(np.float64)
        # Initialise random number generator
        self.rng = np.random.default_rng(SeedSequence(seed))

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

    def get_prob_add_zero(self, y, prob_limit, order):
        l = len(y)
        correlators = self.get_correlators_for_marginal_to_order(y, order)
        return 0.5 * (prob_limit + np.sum(correlators)/(2**l))

    def get_marginal_p_0(self, order):
        self.marginals[''] = 1
        idx0, idx1 = 0, int('1'+'0'*(self.num_qubits-1), 2)
        if order == 0:
            prob_add_zero = 0.5 * self.correlators[idx0]
        else:
            prob_add_zero = 0.5 * (self.correlators[idx0] + self.correlators[idx1])
        self.marginals['0'] = prob_add_zero
        self.marginals['1'] = self.marginals[''] - prob_add_zero
    
    def get_XEB(self, k: int) -> np.float64:
        mask = self.order_arr <= k
        mask[0] = False
        correlators_upto_order_k = mask * self.correlators
        return np.sum(square_mod(correlators_upto_order_k))

    def get_XEBs(self, N: int) -> np.ndarray:
        orders = np.arange(N + 1)
        return np.array([self.get_XEB(order) for order in orders])  
