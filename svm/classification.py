from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

from scipy import interp
import numpy as np
import matplotlib.pyplot as plt


from dataset.dataset import get_classification_dataset
from utils import print_errors

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
except ImportError:
    from sklearn.lda import LDA

try:
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
except ImportError:
    from sklearn.qda import QDA


def k_fold(reg, x_train, y_train):
    kf = KFold(n_splits=5)
    kf.get_n_splits(x_train)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train_index, test_index in kf.split(x_train):
        _x_train, _x_test = x_train.values[train_index, :], x_train.values[test_index, :]
        _y_train, _y_test = y_train[train_index],    y_train[test_index]

        probas_ = reg.fit(_x_train, _y_train).predict_proba(_x_test)

        fpr, tpr, thresholds = roc_curve(_y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    return reg


def run_tree_based_classification():
    x_train, x_test, y_train, y_test = get_classification_dataset(0.3)

    reg = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    reg = k_fold(reg, x_train, y_train)
    print_errors(reg, x_train, y_train, x_test, y_test, msg='Random Forest Classifier', prf=True)

    reg = DecisionTreeClassifier()
    reg = k_fold(reg, x_train, y_train)
    print_errors(reg, x_train, y_train, x_test, y_test, msg='Decision Tree Classifier', prf=True)

    reg = SVC(gamma='scale', probability=True)
    reg = k_fold(reg, x_train, y_train)
    print_errors(reg, x_train, y_train, x_test, y_test, msg='Support Vector Classifier (SVC)', prf=True)

    plt.show()
