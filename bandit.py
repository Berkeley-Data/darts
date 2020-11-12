import pandas as pd
import numpy as np

class Bandit:
    '''
    This class defines a multi-arm bandit under a delayed feedback scenario. It
    applies common explore/exploit policies modified for delayed feedback,
    and determines arm pull allocations for the next round of the multi-arm 
    bandit.
    Inputs:
        - df [pandas.DataFrame]: Pandas DataFrame containing the historical
          arm pull data. If using a UCB1 policy, only provide data for the most
          recent round of arm pulls. Else use all arm pulls.
        - arm_col [string]: string defining the column in df that contains the
          identifiers for the arms being pulled.
        - reward_col [string]: string defining the column in df that contains
          the rewards defined by your usecase.
        - policy [string]: string defining which explore/expolit policy to use.
          Options include:
            - Bayes Upper Confidence Bound ("Bayes_UCB"); default
            - UCB1 ("UCB1")
            - Epsilon Greedy ("epsilon_greedy")
        - t [int: 1 <= t <= inf; optional]: integer defining timestep. Only
          used when applying the UCB1 policy. If not provided and UCB1 policy
          is used, will be set to 1 by default.
        - ucb_scale [float: 0 < ucb_scale <= inf; optional]: defines the number
          of standard deviations to use for the upper confidence bound.
          Defaults to 1.96 for a 95% confidence interval. Only used for the
          "BayesUCB" policy.
        - epsilon [float: 0.0 <= epsilon <= 1.0; optional]: Defines how often
          the Epsilon Greedy policy explores over exploiting the best known
          arm. A value of 0.0 means the algorithm will always choose the best
          performing arm. A value of 1.0 means the best performing arm will 
          never be used. The default value is 0.1.
        - greed_factor [float: -inf < greed_factor < inf; optional]: factor for
          determining how greedy allocations should be. Default value is 1,
          which allocates arm pulls based on normalizing explore/exploit scores.
          Negative values favor poorly performing arms. Positive values favor
          well performing arms. A value of 0 will evenly allocate across all 
          arms regardless of explore/exploit policy. 
    Outputs:
        - This class does not explicity output anything. 

    Usage:
        - Initialize instance of Bandit
        - Call make_allocs()
        - Access allocations by setting a variable equal to the make_allocs
          call or by accessing the .allocs property after calling make_allocs()
    '''


    def __init__(self, df, arm_col, reward_col,
                 policy='Bayes_UCB', t = 1, ucb_scale = 1.96,
                 epsilon = 0.1, greed_factor = 1):
        self.df = df
        self.arm_col = arm_col
        self.reward_col = reward_col
        self.policy = policy
        self.ucb_scale = ucb_scale
        self.t = t
        self.epsilon = epsilon
        self.greed_factor = greed_factor
        self._allocs = None

    def epsilon_greedy_policy(self):
        '''
        Applies a modified Epsilon Greedy Policy for delayed feedback regime.
        Algo from https://arxiv.org/pdf/1902.08593.pdf
        '''
        K = len(self._allocs[self.arm_col].unique()) # number of arms
        expr = self.epsilon/(K-1)
        expt = 1 - self.epsilon
        best_model_index = np.argmax(self._allocs['mean'].values)
        score_vec = [expr if i != best_model_index else expt for i in range(K)]
        return np.array(score_vec)

    def ucb1_policy(self):
        '''
        Applies the UCB1 policy to calculate recommendations.
        '''
        return self._allocs['mean'] + \
               np.sqrt((2 * np.log10(self.t)) / \
                        self._allocs['count'])

    def bayes_ucb_policy(self):
        '''
        Applies the Bayes UCB policy to calculate recommendations.
        '''
        return self._allocs['mean'] + \
               (self.ucb_scale * self._allocs['std'] / \
                np.sqrt(self._allocs['count']))

    def compute_stats(self):
        '''
        Calculates the rewards stats this timestep.
        '''
        self._allocs = self.df[[self.arm_col,self.reward_col]].groupby(self.arm_col)\
                          .agg({self.reward_col:['mean','count','std']})

    def apply_policy(self):
        '''
        Applies policy determined on object initialization.
        '''
        if self.policy == 'UCB1':
            self._allocs['score'] = self.ucb1_policy()
        elif self.policy == 'Bayes_UCB':
            self._allocs['score'] = self.bayes_ucb_policy()
        elif self.policy == 'epsilon_greedy':
            self._allocs['score'] = self.epsilon_greedy_policy()
        else:
            raise ValueError(f'Policy {self.policy} not implemented.\
                               Use ["epsilon_greedy","UCB1","BayesUCB"].')

    def apply_allocation_weights(self):
        '''
        Applies an exponential factor to the scores calculated by the chosen
        policy. Default is 1. Negative weights will favor 
        '''
        self._allocs['exp_score'] = self._allocs['score']**self.greed_factor
        self._allocs['allocation'] = self._allocs['exp_score']/np.sum(self._allocs['exp_score'])

    def make_allocs(self):
        '''
        Wrapper for calculating rewards and applying a policy.
        '''
        self.compute_stats()
        self.apply_policy()
        self.apply_allocation_weights()
        self._allocs = self._allocs.sort_values('allocation', ascending=False)
        return self.allocs

    @property
    def allocs(self):
        return self._allocs 