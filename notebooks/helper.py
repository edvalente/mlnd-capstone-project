import pandas as pd
import numpy as np

def filter_relevant_features(table, fraction):
    """
    Helper to get features with importance above 0.1 of maximum value
    """
    
    return table[table['mi'] > max(table['mi']) * fraction].iloc[:, 0]

def intersect_features(table1, table2, fraction):
    """
    Helper to get intersection of relevant features from two tables
    """
    
    features_i = filter_relevant_features(table1, fraction)
    features_j = filter_relevant_features(table2, fraction)
    
    return list(set(features_i) & set(features_j))