import numpy as np
import pandas as pd

from IO import getPrintProbString
from Sampler import Sampler


class HybridSampler(Sampler):
    def __init__(self, correlators, order_arr, seed=14122000):
        super().__init__(correlators, order_arr, seed)
    
    def add_marginal(self, outcome: str, pruning_depth: int, order: int, prob_limit: np.float64):
        if len(outcome) == pruning_depth: return
        marginal = self.get_prob_add_zero(
            outcome,
            prob_limit,
            order
        )
        # print(f"Outcome: {outcome}, marginal {marginal}, prob_limit: {prob_limit}")
        self.marginals[outcome + '0'] = marginal if marginal <= prob_limit else prob_limit 
        # print(f"Marginal of {outcome + '0'}: {self.marginals[outcome + '0']}")
        self.marginals[outcome + '1'] = prob_limit - self.marginals[outcome + '0']
        # print(f"Marginal of {outcome + '1'}: {self.marginals[outcome + '1']}")
        self.add_marginal(outcome + '0', pruning_depth, order, self.marginals[outcome + '0'])
        self.add_marginal(outcome + '1', pruning_depth, order, self.marginals[outcome + '1'])

    def learning_marginals(self, pruning_depth: int, order: int):
        if pruning_depth < 0 or pruning_depth > self.num_qubits:
            raise ValueError("pruning_depth should be between 0 and the number of qubits")
        self.get_marginal_p_0(order)
        self.add_marginal('0', pruning_depth, order, self.marginals['0'])
        self.add_marginal('1', pruning_depth, order, self.marginals['1']) 

    def sample_random_circuit_hybrid(self, order, VERBOSE=False):
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

    # Use this method if there has been a learning phase where *all*
    # marginals have been computed since it does not check/compute them,
    # but assumes they are in class variable self.marginals
    def sample_random_circuit_fast(self, order, VERBOSE=False):
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
        return outcome

    def sample_events(self, num_outcomes: int, order: int, sample_method, VERBOSE=False):
        if (order > self.num_qubits):
            raise ValueError("The order must be smaller or equal to " + self.num_qubits)
        prob_of_samples = np.array([])
        for i in range(num_outcomes):
            sample = sample_method(order, VERBOSE)
            prob_of_samples = np.append(prob_of_samples, self.marginals[sample])
            # print(f"Outcome {i} is |{sample}>")
        return prob_of_samples
        
    def get_experimental_Hog(self, order, events=int(1e4)):
        return np.mean(self.sample_events(events, order, sample_method=self.sample_random_circuit_hybrid))

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
