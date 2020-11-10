'''
This script aggregates the results of each of the campaigns that were run on a
given day. The output should be a CSV file indexed on voterbase_id. The file
should have the following columns:
    - voterbase_id
    - line_number or campaign_id
    - undecided 
        - result based on score at beginning of conversation
        - null for not contacted, not reached, or not providing an answer to the
          rating question.
        - -1 for Trump
        - 0 for Undecided
        - +1 for Biden
'''

import pandas as pd
from sys import argv

dfs = []
cols = []
#This assumes you pass the dir you want to save to as first arg
for arg in argv[2:]:
    df = pd.read_csv(arg)
    if dfs == []:
        cols = list(df.columns)
        dfs.append(df)
    else:
        if list(df.columns) != cols:
            raise ValueError('CSV at %s does not have consistent columns. Must have: %s'%(arg,str(cols)))
        else:
            dfs.append(df)
combined_df = pd.concat(dfs)
combined_df.to_csv(argv[1]+'result.csv',index = False)

