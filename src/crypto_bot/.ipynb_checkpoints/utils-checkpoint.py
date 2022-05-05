import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def save_obj(filename, obj):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    outfile = open(filename, 'wb')
    pickle.dump(obj, outfile)
    outfile.close()


def load_obj(filename):
    infile = open(filename, 'rb')
    obj = pickle.load(infile)
    infile.close()
    return obj


def plot_roc(info={}):
    for name, data in info.items():
        target, pred = data
        t_fpr, t_tpr, _ = roc_curve(target, pred)
        t_auc = roc_auc_score(target, pred)
        plt.plot(t_fpr, t_tpr, label='ROC {} (area = {:3f})'.format(name, t_auc))

    plt.legend(loc=4)
    plt.show()
    return