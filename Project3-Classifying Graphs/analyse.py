import matplotlib.pyplot as plt
import numpy as np
from main import *


def analyze_by_k(pos='data/molecules-small.pos', neg='data/molecules-small.neg'):
    k = list(range(1, 20))
    minsup = 2

    res_dt = np.zeros(len(k))
    res_sr = np.zeros(len(k))
    res_rf = np.zeros(len(k))

    for index, i in enumerate(k):
        res_dt[index] = np.mean(train_a_basic_model(pos, neg, i, minsup))
        res_sr[index] = np.mean(sequential_covering_for_rule_learning(pos, neg, i, minsup))
        res_rf[index] = np.mean(another_classifier(pos, neg, i, minsup))

    plt.figure()
    plt.plot(k, res_dt, marker='s', markersize=8, label="Decision tree", color="C0")
    plt.plot(k, res_sr, marker='^', markersize=7, label="Sequential rule", color="C1", alpha=0.7)
    plt.plot(k, res_rf, marker='o', markersize=7, label="Random forest", color="C2", alpha=0.7)
    plt.legend()
    plt.xlabel("Number of top frequent sequences")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with the evolution")
    plt.ylim(top=1.1, bottom=0)
    plt.grid()
    plt.savefig("accuracy_k.svg")
    plt.show()


def analyze_by_minsup(pos='data/molecules-small.pos', neg='data/molecules-small.neg'):
    k = 7
    minsup = list(range(15, 20))

    res_dt = np.zeros(len(minsup))
    res_sr = np.zeros(len(minsup))
    res_rf = np.zeros(len(minsup))
    res_svc = np.zeros(len(minsup))

    for index, i in enumerate(minsup):
        res_dt[index] = np.mean(train_a_basic_model(pos, neg, k, i))
        res_sr[index] = np.mean(sequential_covering_for_rule_learning(pos, neg, k, i))
        res_rf[index] = np.mean(another_classifier(pos, neg, k, i, 'svm'))
        res_svc[index] = np.mean(another_classifier(pos, neg, k, i, 'rf'))

    np.save('accuracy_minsup_medium.npy', [[k, minsup], res_dt, res_sr, res_rf, res_svc])


def graph_from_minsup(filename):
    [k, minsup], res_dt, res_sr, res_rf, res_svc = np.load(filename, allow_pickle=True)

    plt.figure()
    plt.plot(minsup, res_dt, marker='s', markersize=8, label="Decision tree", color="C0")
    plt.plot(minsup, res_sr, marker='^', markersize=7, label="Sequential rule", color="C1", alpha=0.7)
    plt.plot(minsup, res_rf, marker='o', markersize=7, label="Random forest", color="C2", alpha=0.7)
    plt.plot(minsup, res_svc, marker='o', markersize=7, label="SVC", color="C3", alpha=0.7)
    plt.legend()
    plt.xlabel("Number of top frequent sequences")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with the evolution of the minimum support")
    plt.ylim(top=1.1, bottom=0)
    plt.xlim(left=0)
    plt.grid()
    plt.savefig("accuracy_minsup.svg")
    plt.show()


if __name__ == "__main__":
    # analyze_by_k()
    # analyze_by_minsup()
    graph_from_minsup("accuracy_minsup_medium.npy")
