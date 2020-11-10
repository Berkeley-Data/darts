'''
This file will take the aggregated results CSV and aggregated predictions CSV
and join the two into one file to be used in calculating rewards.
The output of this file will be a CSV as well as a returned dataframe.
The output will have the following columns
    - voterbase_id
    - line_number or campaign_id
    - model_id
    - undecided_result
    - undecided_label
    - trump_prob
    - undecided_prob
    - biden_prob

This is a One-to-Many join. The results file will have one record for each
voterbase_id and the predictions file will have num_models records for each
voterbase_id.
'''