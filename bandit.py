import pandas as pd
import numpy as np

class Bandit:
    def __init__(self, df, policy='BayesUCB', t = 1, ucb_scale = 2.0, epsilon = 0.1):
        self.df = df
        self.policy = policy
        self.ucb_scale = ucb_scale
        self.t = t
        self.epsilon = epsilon
        self._recs = None

    def epsilon_greedy_policy(self):
        '''
        Applies a modified Epsilon Greedy Policy for delayed feedback regime.
        Algo from https://arxiv.org/pdf/1902.08593.pdf
        '''
        K = len(self._recs['model_id'].unique()) # number of arms
        explore = self.epsilon/(K-1)
        exploit = 1 - self.epsilon
        best_model_index = np.argmax(self._recs['mean'])
        score_vec = [explore if i != best_model_index else exploit for i in range(K)]
        return np.array(score_vec)

    def ucb1_policy(self):
        '''
        Applies the UCB1 policy to calculate recommendations.
        '''
        return self._recs['mean'] + \
               np.sqrt((2 * np.log10(self.t)) / \
                        self._recs['count'])

    def bayes_ucb_policy(self):
        '''
        Applies the Bayes UCB policy to calculate recommendations.
        '''
        return self._recs['mean'] + \
               (self.ucb_scale * self._recs['std'] / \
                np.sqrt(self._recs['count']))

    def calculate_rewards(self):
        '''
        Calculates the rewards for this timestep.
        '''
        self._recs = self.df[['model_id','correct']].groupby('model_id')\
                          .agg({'correct':['mean','count','std']})

    def apply_policy(self):
        '''
        Applies policy determined on object initialization.
        '''
        if self.policy == 'UCB1':
            self._recs['score'] = self.ucb1_policy()
        elif self.policy == 'BayesUCB':
            self._recs['score'] = self.bayes_ucb_policy()
        elif self.policy == 'epsilon_greedy':
            self._recs['score'] = self.epsilon_greedy_policy()
        else:
            raise ValueError(f'Policy {self.policy} not implemented.\
                               Use ["epsilon_greedy","UCB1","BayesUCB"].')

    def make_recs(self):
        '''
        Wrapper for calculating rewards and applying a policy.
        '''
        self.calculate_rewards()
        self.apply_policy()
        self._recs = self._recs.sort_values('score', ascending=False)
        return self.recs

    @property
    def recs(self):
        return self._recs 