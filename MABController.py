# General package imports
import pandas as pd

# Project-specific imports 
from bandit import Bandit

class MABController:
    '''
    This class defines an object that controls the flow of the MAB process.
    Instantiation inputs:
        - res_path: Aggregated CSV of the results from a given calling round. 
                    Should have columns for voterbase ID and the -1, 0, 1 score
                    for Trump, Undecided, Biden.
        - prd_path: Aggregated CSV of the predictions from the previous round of 
                    inference. Should have columns for voterbase_id and two
                    columns for each model that has the label and probability.
        - rew_path: CSV file indicating the cumulative rewards each model got at 
                    each time-step t of the process. This file should have a row
                    for each t and a column for each model's reward at t. 
    '''

    def __init__(self, res_path, prd_path, rew_path = None):
        self.results = pd.read_csv(res_path)
        self.predictions = pd.read_csv(prd_path)
        self.model_ids = self.predictions['model_id'].unique
        if not rew_path:
            model_cols = ['t'] + [f'{model_id}_reward' for model_id in self.model_ids]
            init_data = {}
            for col in model_cols:
                init_data[col] = [0]
            self.t = 1
            self.history = pd.DataFrame(init_data)
        else:
            self.history = pd.read_csv(rew_path)
            self.t = max(self.history['t']) + 1
        self._next_allocation = None

    @property
    def next_allocation(self):
        return self._next_allocation

    @next_allocation.setter
    def next_allocation(self,policy_name='BayesUCB',ucb_scale=2.0):
        '''
        Trains a Bayesian Multi-armed bandit based on the history 
        of rewards for each model. The MAB computes the next allocation
        Inputs:
            self.history
            ucb_scale: upper confidence bound scaling parameter for Bayesian
                       UCB learning policy
        Outputs:
            sets self._next_allocation
        ''' 
        self.bandit = Bandit(self.predictions,self.t, policy=policy_name,ucb_scale=ucb_scale)
        self.bandit.make_recs()
        self._next_allocation = self.bandit.recs

    def update_history(self):
        '''
        Update the history to include the cumulative rewards from newest round.
        Inputs:
            self.bandit
        Outputs:
            Appends new row of rewards to self.history
        '''
        new_row = {'t':self.bandit.t}
        for row in self.bandit.recs.iterrows():
            new_row[f'{row["model_id"]}_reward':row['count']]

        self.history = self.history.append(new_row, ignore_index=True)

     