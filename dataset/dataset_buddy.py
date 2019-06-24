from os import path
import pandas as pd

col_names = [
    'Sports',
    'Religious',
    'Nature',
    'Theatre',
    'Shopping',
    'Picnic',

    'User Id',
]


def get_clustering_dataset(feature_set=None):
    data_set = pd.read_csv(path.join("dataset", "buddymove_holidayiq.csv"))

    if feature_set is None:
        feature_set = col_names[:-1]

    x = data_set[feature_set]  # Features
    y = data_set[col_names[-1]]  # Target variable
    return x, y

