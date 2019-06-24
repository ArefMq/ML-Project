from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

col_names = [
   'fixed acidity',
   'volatile acidity',
   'citric acid',
   'residual sugar',
   'chlorides',
   'free sulfur dioxide',
   'total sulfur dioxide',
   'density',
   'pH',
   'sulphates',
   'alcohol',

   'quality'
]

x_reg = y_reg = x_cla = y = y_cla = None
last_feature_set = None


def load_regression_dataset(feature_set=None):
    data_set = pd.read_csv(path.join("dataset", "winequality-white.csv"), header=None, names=col_names)

    if feature_set is None:
        feature_set = col_names[:-1]

    x = data_set[feature_set]  # Features
    y = data_set.quality  # Target variable
    return x, y


def load_classification_dataset(feature_set=None):
    dataset_a = pd.read_csv(path.join("dataset", "winequality-white.csv"), header=None, names=col_names)
    dataset_b = pd.read_csv(path.join("dataset", "winequality-red.csv"), header=None, names=col_names)

    if feature_set is None:
        feature_set = col_names[:-1]

    x = pd.concat([
        dataset_a[feature_set],
        dataset_b[feature_set]
    ])
    y = np.concatenate([np.array([1] * dataset_a.shape[0]),
                        np.array([0] * dataset_b.shape[0])])
    return x, y


def get_regression_dataset(test_size=0.25, random_state=None, feature_set=None):
    global x_reg, y_reg, last_feature_set
    if x_reg is None or y_reg is None or last_feature_set != feature_set:
        x_reg, y_reg = load_regression_dataset(feature_set)
        last_feature_set = feature_set

    x_train, x_test, y_train, y_test = train_test_split(x_reg, y_reg, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def get_classification_dataset(test_size=0.25, random_state=None, feature_set=None):
    global x_cla, y_cla, last_feature_set
    if x_cla is None or y_cla is None or last_feature_set != feature_set:
        x_cla, y_cla = load_classification_dataset(feature_set)
        last_feature_set = feature_set

    x_train, x_test, y_train, y_test = train_test_split(x_cla, y_cla, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test
