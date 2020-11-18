# DARTS
Dynamic and Responsive Targeting System (DARTS) is a python package for allocating targets from a multiple pools leveraging a multi-arm bandit modified for delayed feedback scenarios. Instead of requiring an online implementation like a typical multi-arm bandit, DARTS allows the user to batch process results, which greatly extends is usefulness in the real world. Check out our Case Study (LINK TBD) to see how DARTS was used to help People's Action target persuadable voters in the 2020 U.S. Presidential Election. 

## Installation
Run the following to install DARTS:
```python
pip install darts-berkeley
```

## Usage
### First Round Allocation
In the first round of allocation, all you need to have set up is a pool of targets and the sub-pools they are derived from. The initial allocation should be an even distribution across sub-pools. This can be calculated on the fly, or manually in simple cases.
```python
# minimum necessary imports
import pandas as pd
from darts import Allocator

# load in the data for the target pool
target_pool = pd.read_csv('path/to/target_pool.csv')

# Specify the columns containing the sub-pool ids.
pool_id_column = 'model_id'

# Specify the column that ranks targets for each pool
score_column = 'probability'

# Specify the column that indicates the id of an individual target
target_column = 'voter_id'

# Specify an even distribution across the sub-pools. This can be done
# using an automated method, but here is a manual example
allocations = {
    		'pool_1' : 0.25,
		'pool_2' : 0.25,
		'pool_3' : 0.25,
		'pool_4' : 0.25
		}

# Specify the number of targets to pull form your pool.
n = 100

# set up allocator and allocate targets
allocator = Allocator(allocations, n, target_pool, pool_id_column,
		  score_column, target_column, strategy='round-robin',
		  order='best')

targets = allocator.allocate_pool()
```
### Subsequent Rounds
The following example assumes that at least one round has been completed and rewards have been calculated for each arm.
```python
# minimum necessary imports
import pandas as pd
from darts import Bandit, Allocator

# load in the data with rewards applied following your own reward logic
results = pd.read_csv('path/to/results.csv')
arm_column = 'pool_id'
reward_column = 'rewards'

# set up a multi-arm bandit and calculate allocations to each arm.
bandit = Bandit(results, arm_column, reward_column, policy = 'Bayes_UCB')
allocations_df = bandit.make_allocs().set_index('arm_column')
allocations = allocations_df['allocation'].to_dict()

# load in the data for the target pool
target_pool = pd.read_csv('path/to/target_pool.csv')

# Specify the columns containing the pool ids. The values in this column must
# be the same as the arms used for the multi-arm bandit.
pool_id_column = 'pool_id'

# Specify the column that ranks targets for each pool
score_column = 'probability'

# Specify the column that indicates the id of an individual target
target_column = 'target_id'

# Specify the number of targets to pull form your pool.
n = 100

# set up allocator and allocate targets
allocator = Allocator(allocations, n, target_pool, pool_id_column,
		  score_column, target_column, strategy='round-robin',
		  order='best')

targets = allocator.allocate_pool()
```
## Modules

The `darts` package is comprised of two modules, `Bandit` and `Allocator`. 

The `Bandit` module is a configurable multi-arm bandit modified for delayed feedback scenarios, scenarios where multiple arm pulls happen before rewards are seen for a given arm. Several different explore/exploit algorithms can be selected depending on the domain and use-case. This object provides the relative allocations for the next set of arm pulls. 

The `Allocator` module can use the relative allocations defined by the the `Bandit` object or those manually specified by the user. With the relative allocations, the `Allocator` object will distribute targets from the specified pools according to strategies defined by the user.

### Bandit
This class defines a multi-arm bandit under a delayed feedback scenario. It applies common explore/exploit policies modified for delayed feedback, and determines arm pull allocations for the next round of the multi-arm bandit.
#### Parameters
`df [pandas.DataFrame]`: Pandas DataFrame containing the historical arm pull data. If using a UCB1 policy, only provide data for the most recent round of arm pulls. Else use all arm pulls.

`arm_col [string]`: string defining the column in `df` that contains the identifiers for the arms being pulled.

`reward_col [string]`: string defining the column in `df` that contains the rewards defined by your use-case.

`policy [string]`: string defining which explore/expolit policy to use. Options include:
- Bayes Upper Confidence Bound (`"Bayes_UCB"`); default
- UCB1 (`"UCB1"`)
- Epsilon Greedy (`"epsilon_greedy"`)

`t [int: 1 <= t <= inf; optional]`: integer defining timestep. Only used when applying the UCB1 policy. If not provided and UCB1 policy is used, will be set to 1 by default.

`ucb_scale [float: 0 < ucb_scale <= inf; optional]`: defines the number of standard deviations to use for the upper confidence bound. Defaults to 1.96 for a 95% confidence interval. Only used for the "BayesUCB" policy.

`epsilon [float: 0.0 <= epsilon <= 1.0; optional]`: Defines how often the Epsilon Greedy policy explores over exploiting the best known arm. A value of 0.0 means the algorithm will always choose the best performing arm. A value of 1.0 means the best performing arm will never be used. The default value is 0.1.

`greed_factor [float: -inf < greed_factor < inf; optional]`: factor for determining how greedy allocations should be. Default value is 1, which allocates arm pulls based on normalizing explore/exploit scores. Negative values favor poorly performing arms. Positive values favor well performing arms. A value of 0 will evenly allocate across all arms regardless of explore/exploit policy. 
#### Outputs
This class does not explicitly output anything.

### Allocator
Class that acts as a mechanism to allocate targets from multiple target pools with options for different picking strategies.
#### Parameters
`allocation_distribution [dict]`: Dictionary defining the target pool distribution. Keys are strings representing the id of each target pool and values are floats representing the percentage of targets to allocate from a given pool. Sum of values must add up to 1.0.

`num_allocations [int]`: Number of total targets to allocation across all pools. If this number is greater than the number of targets in `pool`, all of the targets will be allocated.

`pool [pandas.DataFrame]`: Pandas DataFrame containing the targets in each target pool. At a minimum, this object must have columns that contain target ids, pool ids, and scores for each target/pool combination.

`pool_id_col [string]`: String defining the column in `pool` that contains the identifiers for each target pool. The ids in this column must match the keys in `allocation_distribution`.

`target_id_col [string]`: String defining the column in `pool` that contains the identifiers for each target.

`score_col [string]`: String defining the column in `pool` that contains the scores for each target/pool combination. Scores can be on different scales if needed for a given pool (e.g., different modeling techniques) as relative scoring only matters within pool. If not using models to generate target scores, use binary indicators.

`strategy [string; optional]`: String defining how targets are picked from each target pool. Options include:
- `round-robin` [default]: In the round-robin strategy, a single target is chosen from each pool sequentially starting with the pool receiving the least allocations and snaking up and down through all pools. A pool will continue to be included in the round-robin until a pool runs out of its allocations (defined by the allocation distribution and number of total allocations). 
- `greedy`: In the greedy strategy, targets are selected from pools in descending order of the number of allocations for each pool. Instead of selecting one target at a time like in round-robin, each pool will get all of its targets allocated in a batch.
- `altruist`: The altruist strategy is the same as the greedy strategy except the pool with the fewest target allocations contributes targets first.

`order [str; optional]`:  Order dictating which target should be picked from a given pool. Options:
- `random`: picks a random target from a pool.
- `best` [default]: picks the target with the highest score from a pool.
- `worst`: picks the target with the lowest score from a pool.


