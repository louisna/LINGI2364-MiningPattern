import matplotlib.pyplot as plt
import numpy as np
from main import *
from tqdm import tqdm


def analyze_by_k(pos='data/molecules.pos', neg='data/molecules.neg'):
    k = list(range(1, 40))
    minsup = 2200

    res_dt = np.zeros(len(k))
    res_sr = np.zeros(len(k))
    res_rf = np.zeros(len(k))
    res_svc = np.zeros(len(k))

    with open("accuracy_k_big_2200.txt", "a") as fd:
        for index in tqdm(range(len(k))):
            i = k[index]
            res_dt[index] = np.mean(train_a_basic_model(pos, neg, i, minsup))
            res_sr[index] = np.mean(sequential_covering_for_rule_learning(pos, neg, i, minsup))
            res_rf[index] = np.mean(another_classifier(pos, neg, i, minsup, 'rf'))
            res_svc[index] = np.mean(another_classifier(pos, neg, i, minsup, 'svm'))

            fd.write("{} {} {} {} {}\n".format(i, res_dt[index], res_sr[index], res_rf[index], res_svc[index]))
            fd.flush()


def graph_from_k(filename, minsup):
    k, res_dt, res_sr, res_rf, res_svc = read_from_file(filename)

    plt.figure()
    plt.plot(k, res_dt, marker='s', markersize=8, label="Decision tree", color="C0")
    plt.plot(k, res_sr, marker='^', markersize=7, label="Sequential rule", color="C1", alpha=0.7)
    plt.plot(k, res_rf, marker='o', markersize=7, label="Random forest", color="C2", alpha=0.7)
    plt.plot(k, res_svc, marker='o', markersize=7, label="SVC", color="C3", alpha=0.7)
    plt.legend()
    plt.xlabel("Number of top frequent sequences")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with the evolution")
    plt.ylim(top=1.1, bottom=0)
    plt.grid()
    plt.savefig("accuracy_k_medium.svg")
    plt.show()


def analyze_by_minsup(pos='data/molecules.pos', neg='data/molecules.neg'):
    k = 7
    minsup = list(range(1800, 2431, 10))
    minsup.reverse()

    res_dt = np.zeros(len(minsup))
    res_sr = np.zeros(len(minsup))
    res_rf = np.zeros(len(minsup))
    res_svc = np.zeros(len(minsup))

    with open("accuracy_minsup_big_crt.txt", "a") as fd:
        for index in tqdm(range(len(minsup))):
            i = minsup[index]
            res_dt[index] = np.mean(train_a_basic_model(pos, neg, k, i))
            res_sr[index] = np.mean(sequential_covering_for_rule_learning(pos, neg, k, i))
            res_rf[index] = np.mean(another_classifier(pos, neg, k, i, 'rf'))
            res_svc[index] = np.mean(another_classifier(pos, neg, k, i, 'svm'))

            fd.write("{} {} {} {} {}\n".format(i, res_dt[index], res_sr[index], res_rf[index], res_svc[index]))
            fd.flush()

    # np.save('accuracy_minsup_big.npy', [[k, minsup], res_dt, res_sr, res_rf, res_svc])


def read_from_file(filename):
    dt = []
    sr = []
    rf = []
    svc = []
    values = []
    with open(filename, 'r') as fd:
        for line in fd:
            a, b, c, d, e = line.split(" ")
            values.append(float(a))
            dt.append(float(b))
            sr.append(float(c))
            rf.append(float(d))
            svc.append(float(e))
    return values, dt, sr, rf, svc


def graph_from_minsup(filename, k):
    minsup, res_dt, res_sr, res_rf, res_svc = read_from_file(filename)

    print(len(minsup))

    plt.figure()
    plt.plot(minsup, res_dt, marker='s', markersize=8, label="Decision tree", color="C0")
    plt.plot(minsup, res_sr, marker='^', markersize=7, label="Sequential rule", color="C1", alpha=0.7)
    plt.plot(minsup, res_rf, marker='o', markersize=7, label="Random forest", color="C2", alpha=0.7)
    plt.plot(minsup, res_svc, marker='o', markersize=7, label="SVC", color="C3", alpha=0.7)
    plt.legend()
    plt.xlabel("Number of top frequent sequences")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with the evolution of the minimum support")
    plt.ylim(top=0.8, bottom=0.5)
    plt.grid()
    plt.savefig("accuracy_minsup.svg")
    plt.show()


if __name__ == "__main__":
    analyze_by_k()
    # analyze_by_minsup()
    # analyze_by_k()
    # analyze_by_minsup()
    # graph_from_minsup("accuracy_minsup_big_crt.txt", 7)
