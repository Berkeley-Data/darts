from allocation import Allocator
import pandas as pd

# Load in and transform predictions from CSV

preds = None

# We will use a uniform allocation for the initial set up
alloc_dict = {
              'model1': 0.25,
              'model2': 0.25,
              'model3': 0.25,
              'model4': 0.25
             }

#Specify the number of people to allocate
n = 100000

# run the allocation
allocator = Allocator(alloc_dict,n,preds,strategy='round-robin',order='random')
allocations = allocator.allocate_predictions()

