import numpy as np
from scipy.stats import t as t_table
from sklearn.linear_model import LinearRegression

from dataset.dataset import get_regression_dataset, col_names
from utils import print_errors

MAX_ITERATIONS = 1000


def main(feature_set):
    coef_list = []
    for iteration in range(MAX_ITERATIONS):
        print 'iteration: %d\r' % (iteration+1),
        x_train, x_test, y_train, y_test = get_regression_dataset(0.6, feature_set=feature_set)
        # x_train, x_test = x_train[feature_set], x_test[feature_set]
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        coef_list.append(lr.coef_)

    coef_list = np.array(coef_list)
    se = np.std(coef_list, 0) / np.sqrt(MAX_ITERATIONS)
    t = np.mean(coef_list, 0) / se
    pvalue = t_table.sf(np.fabs(t), len(t)-1)*2
    coef_list = np.mean(coef_list, 0)

    print '\n\n{:25s}   {:s}         {:s}  {:s}     {:s}'.format('Field',
                                                                 'COEF',
                                                                 'Standard Error',
                                                                 't-Statistics',
                                                                 'P-value')
    print '================================================================================'
    for values in zip(feature_set, coef_list, se, t, pvalue):
        print '{:25s}   {:3.4f} \t    {:3.4f} \t    {:3.4f} \t  {:3.6f}'.format(*values)
    print '\n'
    print_errors(lr, x_train, y_train.values, x_test, y_test.values, msg='Full Features')


def run_linear_regression():
    print 'Full Feature ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    main(col_names[:-1])

    print 'Selected Feature ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # selected by running feature_selection.py
    selected_features = ['fixed acidity', 'residual sugar', 'free sulfur dioxide', 'pH']
    main(selected_features)
