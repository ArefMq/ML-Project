import numpy as np
import math
from sklearn.metrics import precision_recall_fscore_support


def save_coefs(reg, file_name):
    np.savetxt(file_name, reg.coef_, fmt='%.4f', delimiter='\n')


def RSS(y_pred, y_label):
    return np.sum((y_pred - y_label) ** 2)


def TSS(y_label):
    y_mean = np.mean(y_label, 0)
    return np.sum((y_label - y_mean) ** 2)


def R2(y_pred, y_label):
    return 1.0 - (RSS(y_pred, y_label) / TSS(y_label))


def BIC(y_pred, y_label, num_of_features):
    n = len(y_pred)
    errors = (y_pred - y_label)
    error_mean = np.mean(y_label, 0)
    variance = np.sum((errors - error_mean) ** 2) / n-2
    return (RSS(y_pred, y_label) + math.log(len(y_pred)) * num_of_features * variance) / n


def print_errors(reg, x_train, y_train, x_test=None, y_test=None, msg=None, prf=False):
    def _print_detail(msg, y_pred, y_label, prf):
        y_pred = y_pred.flatten()
        y_label = y_label.flatten()

        print '%s Error:' % msg
        print ' - RSS = %.3f' % (RSS(y_pred, y_label))
        print ' - TSS = %.3f' % (TSS(y_label))
        print ' - R^2 = %.3f' % (R2(y_pred, y_label))
        print ''

        # try:
        if prf:
            y_pred = [0 if i < 0.5 else 1 for i in y_pred]
            y_label = [0 if i < 0.5 else 1 for i in y_label]
            precision, recal, f_score, support = precision_recall_fscore_support(y_pred, y_label, average='macro')
            print ' - Precision: %0.3f\n - Recal: %0.3f\n - F-Score: %0.3f' % (precision, recal, f_score)
            print '---------------------------------------------'
        # except:
        #     pass

    print '\n=============================================\n'
    if msg:
        print msg
    _print_detail('Train', reg.predict(x_train), y_train, prf)
    if x_test is None or y_test is None:
        return

    y_pred = reg.predict(x_test)
    _print_detail('Test', y_pred, y_test, prf)

    return R2(y_pred, y_test)
