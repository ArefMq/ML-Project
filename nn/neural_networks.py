import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, LeaveOneOut

from dataset.dataset import get_classification_dataset
from utils import print_errors

draw_chance = False
EPOCHS = 20
BATCH_SIZE = 128
KERAS_VERBOSITY = 0


def k_fold(model, x_train, y_train, k=5):
    global draw_chance
    if k == -1:
        kf = LeaveOneOut()
    else:
        kf = KFold(n_splits=k)
    kf.get_n_splits(x_train)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train_index, test_index in kf.split(x_train):
        _x_train, _x_test = x_train.values[train_index, :], x_train.values[test_index, :]
        _y_train, _y_test = y_train[train_index], y_train[test_index]

        model.fit(_x_train, _y_train, epochs=EPOCHS, verbose=KERAS_VERBOSITY, batch_size=BATCH_SIZE)
        pro_bas = model.predict(_x_test)

        fpr, tpr, thresholds = roc_curve(_y_test, pro_bas)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    if not draw_chance:
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, label='Chance', alpha=.8)
        draw_chance = True

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, label=r'%s ROC (AUC = %0.2f $\pm$ %0.2f)' % (model.name, mean_auc, std_auc), lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    return model


def model_1():
    model = Sequential(name='model #1')

    model.add(Dense(11, activation='relu', input_dim=11))
    model.add(Dense(11, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def model_2():
    model = Sequential(name='model #2')

    model.add(Dense(11, activation='relu', input_dim=11))
    model.add(Dense(11, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def model_3():
    model = Sequential(name='model #3')

    model.add(Dense(11, activation='relu', input_dim=11))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def model_4():
    model = Sequential(name='model #4')

    model.add(Dense(11, activation='relu', input_dim=11))
    model.add(Dense(11, activation='relu'))
    model.add(Dense(11, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def model_5():
    model = Sequential(name='model #5')

    model.add(Dense(11, activation='sigmoid', input_dim=11))
    model.add(Dense(11, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def main(k):
    x_train, x_test, y_train, y_test = get_classification_dataset(0.3)
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    models = [model_1, model_2, model_3, model_4, model_5]

    for m in models:
        mdl = m()
        k_fold(mdl, x_train, y_train, k)
        print_errors(mdl, x_train, y_train, x_test, y_test, msg=mdl.name, prf=True)

    plt.show()


def run_nn():
    main(k=5)
