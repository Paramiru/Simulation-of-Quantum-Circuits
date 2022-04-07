import numpy as np
import pandas as pd
from Sampler import Sampler

class OnTheFlySampler(Sampler):
    def __init__(self, correlators, order_arr, seed=14122000):
        super().__init__(correlators, order_arr, seed)

    def get_correlators_for_marginal_slow(self, y):
        # Brute force method for the first implementation
        l = len(y)
        # y is a bitstring containing a number in binary
        y = int(y, 2)
        correlators = np.array([])
        for i in range(2**l):
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
        correlators = self.get_correlators_for_marginal_slow(y)
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

    def sample_events(self, num_outcomes: int):
        prob_of_samples = np.array([])
        for i in range(num_outcomes):
            sample = self.sample_random_circuit_slow()
            prob_of_samples = np.append(prob_of_samples, self.marginals[sample])
            print(f"Outcome {i} is |{sample}>")
        return prob_of_samples
        
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
