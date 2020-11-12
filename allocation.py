import json
from math import ceil
import pandas as pd
from random import choice
import warnings

class Allocator:
    '''
    Class that acts as a mechanism to allocate targeting predictions
    from multiple models with options for different picking strategies.
    Configuration parameters:
        - allocation distribution (dict): keys are model ids (str) and values 
                                          are the proportion of targets to 
                                          allocate (float). Sum of values must
                                          add up to 1.0.
        - num_allocations (int): number of vb_voterbase_ids to allocate across
                                 the models. If this number is greater than the
                                 number of ids in the predictions dataframe, the
                                 maximum number of ids in the predictions
                                 dataframe will be used instead.
        - predictions (pd.DataFrame): Pandas DataFrame that contains the
                                      predictions for each model. At a minimum,
                                      this object must have columns:
                                         - vb_voterbase_id (any type)
                                         - model_id (str matching model_ids in
                                                    allocation_distribution)
                                         - probability (float)
                                      If your model only predicts classes and
                                      not probabilities, use 1 for the 
                                      probability column.
        - strategy (str) [optional]: Strategy for how predictions are picked
                                     from the targeting allocation. Values:
                                        - round-robin [default]: starts with 
                                          the model with the least allocation
                                          and snakes through the sequence of 
                                          models until num_allocations reached.
                                        - greedy: picks predictions from the 
                                          model with the highest allocation 
                                          until that model has no more 
                                          allocations left, then from the next
                                          model in decreasing order.
                                        - altruist: picks predictions from the 
                                          model with the least allocation until
                                          that model has no more allocations
                                          left, then from the next model in 
                                          increasing order.
        - order (str) [optional]: Order dictating which prediction should be
                                  picked from a given model. Values:
                                    - random: picks a random prediction from 
                                      the model.
                                    - best: picks the prediction with the
                                      highest probability from the model.
                                    - worst: picks the prediction with the
                                      lowest probability from the model.
    Usage: Object is configured on instantiation. To extract a list of ids
           allocated by the model, call the allocate_predictions method. This
           method has output that can be assigned to a variable or it can be
           ran in place. If ran in place, access the allocations with the 
           .allocations property.
    '''

    def __init__(self, 
                 allocation_distribution,
                 num_allocations,
                 pool,
                 strategy = 'round-robin',
                 order = 'random'):
        self._num_allocations = None
        self.num_allocations = num_allocations
        self._allocation_distribution = None
        self.allocation_distribution = allocation_distribution
        self._pool = None
        self.pool = pool
        self._strategy = None
        self.strategy = strategy
        self._order = None
        self.order = order
        self._allocations = []

    @property
    def allocation_distribution(self):
        return self._allocation_distribution

    @allocation_distribution.setter
    def allocation_distribution(self,allocation_distribution):
        if not isinstance(allocation_distribution,dict):
            raise TypeError('Must pass pool allocations as a dictionary keyed '
                            'by pool ID (str) with values being floats '
                            'representing the allocation percentage.')
        elif len(set([type(k) for k in allocation_distribution.keys()])) > 1:
            raise TypeError('At least one pool id in allocation_distribution '
                             'is not a string. Verify supplied keys.')
        elif len(set([type(v) for v in allocation_distribution.values()])) >1:
            raise TypeError('At least one allocation in allocation_distribution'
                            ' is not a float. Verify supplied allocations.')
        
        sum_of_model_allocations = sum([v for v in allocation_distribution.values()])
        
        if sum_of_model_allocations != 1.0:
            raise ValueError('Sum of pool allocations must equal 1.0 not '
                             f'{sum_of_model_allocations}.')            
        else:
            self._allocation_distribution = allocation_distribution

    @property
    def pool(self):
        return self._pool

    @pool.setter
    def pool(self,obj):
        '''
        Validates the pool passed to the constructor or any updated
        pool after initialization. Sets up the "picked" variable and 
        sorts the DataFrame by model_id and probability.
        '''
        if not isinstance(obj,pd.DataFrame):
            raise TypeError(f'pool must be a pd.DataFrame, not {type(obj)}')
        else:
            self._pool = obj
            self._pool['picked'] = 0
            self._pool.sort_values(['model_id','probability'],
                                           ascending=False,
                                           inplace=True)

    @property
    def num_allocations(self):
        return self._num_allocations

    @num_allocations.setter
    def num_allocations(self,num):
        '''
        Verifies that a valid parameter is set for the number of allocations.
        '''
        if not isinstance(num,int):
            raise TypeError(f'Number of allocations must be an integer not {type(num)}.')
        elif num < 1:
            raise ValueError('Specified number of allocations must be at least 1.')
        else:
            self._num_allocations = num

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        '''
        Checks if strategy is valid and sets it if so.
        '''
        if strategy not in ['round-robin','greedy','altruist']:
            raise ValueError(f'Strategy {strategy} is not valid. Use one of '
                             '["round-robin","greedy","altruist"].')
        else:
            self._strategy = strategy

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        '''
        Checks if order is valid and sets it if so.
        '''
        if order not in ['random','best','worst']:
            raise ValueError(f'Order {order} is not valid. Use one of '
                             '["random","best","worst"].')
        else:
            self._order = order

    @property
    def allocations(self):
        return self._allocations

    def pick_index(self, indices):
        '''
        Returns the index of a prediction to pick based on the order strategy.
        '''
        if self.order == 'random':
            # get random index
            return choice(indices)
        elif self.order == 'best':
            # should be in descending order so this gets best
            return indices[0]
        else:
            # case when worst pred is chosen first
            return indices[-1]

    def pick_id(self,model_id):
        '''
        For a given model, picks an index and extracts the vb_voterase_id at
        that index.
        '''
        # get a pred id from the model per to order strategy
        indices = self.pool[(self.pool['model_id'] == model_id)
                                    & (self.pool['picked'] == 0)].index
        idx = self.pick_index(indices)
        vid = self.pool.at[idx,'vb_voterbase_id']
        return idx, vid

    def allocate_id(self,index,voter_id,model_id):
        '''
        Adds the given vb_voterbase_id to the allocations and updates all rows
        with that id to indicate it has been picked already.
        '''
        #allocate the ID
        self.allocations.append((voter_id,model_id))
        # set picked column for that id
        id_indices = self.pool[self.pool['vb_voterbase_id'] == voter_id].index
        for i in id_indices:
            self.pool.at[i,'picked'] = 1

    def make_n_allocations_from_model(self,n,model_id):
        '''
        Wrapper to make multiple allocations for a model at once.
        '''
        if not n >= 1:
            raise ValueError('Number of allocations to make is invalid. \
                              Must be >= 1')
        made = 0
        while made < n:
            idx, v_id = self.pick_id(model_id)
            self.allocate_id(idx,v_id,model_id)
            made += 1

    def allocate_pool(self):
        '''
        Performs checks to ensure appropriate amount of allocations are made.
        Then makes the allocations based on the strategy set for the mechanism.
        Allocations are made to the list object associated with this object 
        instance.
        Also returns self.allocations for direct access when called.
        '''
        num_models = len(self.pool['model_id'].unique())
        total_preds = self.pool.shape[0]//num_models
        if total_preds < self.num_allocations:
            warnings.warn('Specified number of allocations, '
                          f'{self.num_allocations}, is greater than the '
                          'number of pool provided. Only allocating '
                          f'{total_preds} pool.')
        else:
            total_preds = self.num_allocations

        # generate the number of pool to make for each model
        models = []
        for model,allocation in self.allocation_distribution.items():
            models.append({'id':model,
                           'count':ceil(total_preds*allocation),
                           'allocated':0})

        # generate order of model choice, model with fewest preds goes first
        models.sort(key=lambda x:x['count'])

        if self._strategy == 'round-robin':
            allocated = 0
            current_model_index = 0
            while allocated < total_preds:
                model = models[current_model_index]
                if model['allocated'] < model['count']:
                    self.make_n_allocations_from_model(1,model['id'])
                    allocated += 1
                    model['allocated'] += 1
                # because we are going in round-robin order, reverse the next time
                if current_model_index == num_models - 1:
                    models.reverse()
                    current_model_index = 0
                else:
                    current_model_index += 1
        elif self._strategy == 'greedy':
            # greedy strat, pick from model with greatest allocation first
            remaining_allocations = total_preds - len(self.allocations)
            for model in reversed(models):
                n = min(model['count'],remaining_allocations)
                if n >= 1:
                    self.make_n_allocations_from_model(n,model['id'])
                    remaining_allocations -= n
        else:
            # altruist strat, pick from model with smallest allocation first
            remaining_allocations = total_preds - len(self.allocations)
            for model in models:
                n = min(model['count'],remaining_allocations)
                if n >= 1:
                    self.make_n_allocations_from_model(n,model['id'])
                    remaining_allocations -= n

        return self.allocations

if __name__ == '__main__':
    test_preds = pd.read_csv('test_files/test_predictions1.csv')
    with open('test_files/test_allocation1.json', 'r') as read_file:
        test_alloc_dist = json.load(read_file)
    allocator = Allocator(test_alloc_dist,4,test_preds,strategy='round-robin',order='random')
    a = allocator.allocate_pool()
    print(a)