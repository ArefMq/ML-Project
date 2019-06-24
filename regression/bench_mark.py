from os import path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from dataset.dataset import get_regression_dataset
from utils import save_coefs, print_errors


def run_bench_mark():
    x_train, x_test, y_train, y_test = get_regression_dataset(random_state=0)

    print 'Linear Regression ==========================='
    logreg = LinearRegression()
    logreg.fit(x_train, y_train)
    print_errors(logreg, x_train, y_train, x_test, y_test)
    # save_coefs(logreg, path.join('regression', 'results', 'linear.coef'))

    print 'Ridge Regression ============================'
    clf = Ridge(alpha=1.0)
    clf.fit(x_train, y_train)
    print_errors(clf, x_train, y_train, x_test, y_test)
    # save_coefs(clf, path.join('regression', 'results', 'ridge.coef'))

    print 'Lasso Regression ============================'
    lso = Lasso(alpha=0.1)
    lso.fit(x_train, y_train)
    print_errors(lso, x_train, y_train, x_test, y_test)
    # save_coefs(lso, path.join('regression', 'results', 'lasso.coef'))

    print 'ElasticNet Regression ======================='
    eln = ElasticNet(random_state=0)
    eln.fit(x_train, y_train)
    print_errors(eln, x_train, y_train, x_test, y_test)
    # save_coefs(eln, path.join('regression', 'results', 'elastic_net.coef'))

