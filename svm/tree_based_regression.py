from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
import numpy as np

from dataset.dataset import get_regression_dataset
from utils import print_errors, R2


MAX_ITERATIONS = 1000


def main(tree_depth):
    x_train, x_test, y_train, y_test = get_regression_dataset(0.01)

    kf = KFold(n_splits=5)
    kf.get_n_splits(x_train)
    r2s = []
    for train_index, test_index in kf.split(x_train):
        _x_train, _x_test = x_train.values[train_index, :], x_train.values[test_index, :]
        _y_train, _y_test = y_train.values[train_index], y_train.values[test_index]

        r2s.append(tree_based_reg(tree_depth, _x_train, _x_test, _y_train, _y_test))
    return np.average(r2s)


def tree_based_reg(tree_depth, x_train, x_test, y_train, y_test):
    regr = DecisionTreeRegressor(max_depth=tree_depth)
    regr.fit(x_train, y_train)
    return R2(regr.predict(x_test), y_test)


def run_tree_based_regression():
    best_r2 = None
    best_depth = None

    for i in range(2, 20):
        r2 = main(i)
        if best_r2 is None or r2 < best_r2:
            best_r2 = r2
            best_depth = i

    # re run the regression on the best depth
    print 'best depth is %d' % best_depth
    print 'results:'
    x_train, x_test, y_train, y_test = get_regression_dataset()
    regr = DecisionTreeRegressor(max_depth=best_depth)
    regr.fit(x_train, y_train)
    print_errors(regr, x_train, y_train, x_test, y_test, msg='Tree-Based Regression')
