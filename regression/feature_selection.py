from sklearn.linear_model import LinearRegression

from dataset.dataset import get_regression_dataset, col_names
from utils import R2, BIC, RSS, TSS

FEATURE_LIST = col_names[:-1]


def get_feature_list(counter):
    selected = []
    valid_features = {i: 2 ** i for i in range(len(FEATURE_LIST))}
    for f, v in valid_features.items():
        if v & counter:
            selected.append(f)
    return [FEATURE_LIST[t] for t in selected]


def run_feature_selection():
    x_train, x_test, y_train, y_test = get_regression_dataset(0.6)
    config_table = {}

    for counter in range(1, 2 ** len(FEATURE_LIST)):
        subset_features = get_feature_list(counter)
        subset_x = x_train[subset_features]

        lr = LinearRegression()
        lr.fit(subset_x, y_train)

        subset_x = x_test[subset_features]
        y_pred = lr.predict(subset_x)
        r2 = R2(y_pred, y_test.values)
        bic = BIC(y_pred, y_test.values, len(subset_features))

        if r2 > 0:
            config_table[tuple(subset_features)] = (r2, bic)

    iter = 0
    for key, value in sorted(config_table.iteritems(), key=lambda (k, v): (v[1], k)):
        print '%s \t (R2 = %0.3f | BIC = %0.3f)' % (key, value[0], value[1])

        if iter < 10:
            iter += 1
        else:
            break
