'''
Whenever inference is run on the set of models, each model should output a file
named after the model's model_id that contains the following:
    - voterbase_id
    - predicted undecided label
    - predicted probability of class -1
    - predicted probability of class 0
    - predicted probability of class 1

This script aggregates each of those files and outputs a CSV indexed on
voterbase_id that contains the following columns:
    - voterbase_id
    - model_id
    - undecided_label
    - trump_prob
    - undecided_prob
    - biden_prob
'''