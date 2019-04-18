import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def custom_f1_score(y_pred, y_true):
    labels = y_true.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y_pred]
    return 'f1_score', f1_score(labels, y_bin)

def custom_fbeta_score(y_pred, y_true):
    labels = y_true.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y_pred]
    return 'fbeta_score', fbeta_score(labels, y_bin, beta=0.5)

def custom_precision_score(y_pred, y_true):
    labels = y_true.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y_pred]
    return 'precision_score', precision_score(labels, y_bin)

def custom_accuracy_score(y_pred, y_true):
    labels = y_true.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y_pred]
    return 'accuracy_score', accuracy_score(labels, y_bin)

def preprocess_minimal_data(df):
    """
    Preprocesses minimal data into train and test sets
    for supervised and unsupervised learning.
    Unsupervised learning data has log transformation and needs
    normalization to be applied.
    """
    # Copy data for clustering
    cluster_df = df.copy()
    cluster_df.loc[:, 'days_to_appointment'] = np.log(cluster_df.loc[:, 'days_to_appointment'])
    normalizer = MinMaxScaler().fit(cluster_df)
    
    # Split dataset
    X = df.drop('no_show', axis=1)
    y = df.loc[:, 'no_show']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Split cluster data
    cluster_train = cluster_df.loc[X_train.index.values]
    cluster_test = cluster_df.loc[X_test.index.values]
    
    return (X_train, X_test, y_train, y_test, cluster_train, cluster_test, normalizer)

def filter_relevant_features(table, fraction=0.1):
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

def recommend(data, centers, rec_index, input_indexes, label_index):
    
    individuals_distances = distances(data, centers, input_indexes)
    valid_clusters = [i for i, j in enumerate(centers) if j[label_index] > 0.75]
    recommended_values = {k: v[rec_index] for k, v in enumerate(centers) if k in valid_clusters}
    
    recommendations = []
    for individual_distance in individuals_distances:
        valid_distances = {k: v for k, v in individual_distance.items() if k in valid_clusters}
        recommended_cluster = min(valid_distances, key=valid_distances.get)
        recommended_value = int(recommended_values[recommended_cluster])
        recommendations.append(recommended_value)
        
    return np.array(recommendations)