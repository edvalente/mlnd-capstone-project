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

def calc_distance(a, b):
    """
    Calculates euclidean distance between two points
    """
    return np.sqrt((np.array(a) - np.array(b))**2).sum()

def calc_distances(a, centers, indexes):
    """
    Calculates euclidean distances between one point and several
    """
    return {i: calc_distance(a[indexes], j[indexes]) for i, j in enumerate(centers)}

def distances(individuals, centers, indexes):
    """
    Calculates euclidean distances between two arrays of points
    """
    return np.array([calc_distances(np.array(j), centers, indexes) for i, j in individuals.iterrows()])