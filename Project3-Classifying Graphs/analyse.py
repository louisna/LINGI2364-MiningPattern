import matplotlib.pyplot as plt
import numpy as np
from main import *
from tqdm import tqdm


def analyze_by_k(pos='data/molecules.pos', neg='data/molecules.neg'):
    k = list(range(1, 150, 10))
    minsup = 1200

    res_dt = np.zeros((len(k), 8))
    res_sr = np.zeros((len(k), 8))
    res_rf = np.zeros((len(k), 8))
    res_svc = np.zeros((len(k), 8))
    res_knn = np.zeros((len(k), 8))

    with open("accuracy_k_big_1200_balanced.txt", "a+") as fd:
        for index in tqdm(range(len(k))):
            i = k[index]
            res_dt[index] = train_a_basic_model(pos, neg, i, minsup)
            res_sr[index] = sequential_covering_for_rule_learning(pos, neg, i, minsup)
            res_rf[index] = another_classifier(pos, neg, i, minsup, 'rf')
            res_svc[index] = another_classifier(pos, neg, i, minsup, 'svm')
            res_knn[index] = another_classifier(pos, neg, i, minsup, 'knn')
            fd.write(str(i) + "\n")
            for array in [res_dt[index], res_sr[index], res_rf[index], res_svc[index], res_knn[index]]:
                a = ""
                for v in array:
                    a += str(v) + " "
                a += '\n'
                fd.write(a)
            fd.write('\n')  # Again for cleaning
            fd.flush()


def graph_from_k(filename):
    k, res_dt, res_sr, res_rf, res_svc, res_knn = read_from_file_global(filename)

    res_dt = [np.mean(i) for i in res_dt]
    res_sr = [np.mean(i) for i in res_sr]
    res_rf = [np.mean(i) for i in res_rf]
    res_svc = [np.mean(i) for i in res_svc]
    res_knn = [np.mean(i) for i in res_knn]

    plt.figure()
    plt.plot(k, res_dt, marker='s', markersize=8, label="Decision tree", color="C1", alpha=0.7)
    plt.plot(k, res_sr, marker='^', markersize=7, label="Sequential rule", color="C0", alpha=0.7)
    plt.plot(k, res_rf, marker='o', markersize=7, label="Random forest", color="C2", alpha=0.7)
    plt.plot(k, res_svc, marker='x', markersize=7, label="SVC", color="C3", alpha=0.7)
    plt.plot(k, res_knn, marker='.', markersize=7, label="kNN", color="C4", alpha=0.7)
    plt.legend()
    plt.xlabel("Number of top frequent sequences (k)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with the evolution of k for the medium dataset")
    plt.ylim(top=0.8, bottom=0.5)
    plt.grid()
    plt.savefig("accuracy_k_medium_50.svg")
    plt.show()


def analyze_by_minsup(pos='data/molecules.pos', neg='data/molecules.neg'):
    k = 30
    minsup = list(range(1000, 2000, 30))

    res_dt = np.zeros((len(minsup), 8))
    res_sr = np.zeros((len(minsup), 8))
    res_rf = np.zeros((len(minsup), 8))
    res_svc = np.zeros((len(minsup), 8))
    res_knn = np.zeros((len(minsup), 8))

    with open("accuracy_minsup_big_30_balanced.txt", "a+") as fd:
        for index in tqdm(range(len(minsup))):
            i = minsup[index]
            res_dt[index] = train_a_basic_model(pos, neg, k, i)
            res_sr[index] = sequential_covering_for_rule_learning(pos, neg, k, i)
            res_rf[index] = another_classifier(pos, neg, k, i, 'rf')
            res_svc[index] = another_classifier(pos, neg, k, i, 'svm')
            res_knn[index] = another_classifier(pos, neg, k, i, 'knn')

            fd.write(str(i) + "\n")
            for array in [res_dt[index], res_sr[index], res_rf[index], res_svc[index], res_knn[index]]:
                a = ""
                for v in array:
                    a += str(v) + " "
                a += '\n'
                fd.write(a)
            fd.write('\n')  # Again for cleaning
            fd.flush()


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


def read_from_file_global(filename):
    dt = []
    sr = []
    rf = []
    svc = []
    knn = []
    values = []
    with open(filename, 'r') as fd:
        lines = fd.readlines()
        index = 0
        while index < len(lines):
            values.append(int(lines[index]))
            dt.append([float(i) for i in lines[index + 1].split(" ")[:-1]])
            sr.append([float(i) for i in lines[index + 2].split(" ")[:-1]])
            rf.append([float(i) for i in lines[index + 3].split(" ")[:-1]])
            svc.append([float(i) for i in lines[index + 4].split(" ")[:-1]])
            knn.append([float(i) for i in lines[index + 5].split(" ")[:-1]])
            index += 7
    return values, np.array(dt), np.array(sr), np.array(rf), np.array(svc), np.array(knn),




def graph_from_minsup(filename, k):
    # minsup, res_dt, res_sr, res_rf, res_svc = read_from_file(filename)
    minsup, res_dt, res_sr, res_rf, res_svc, res_knn = read_from_file_global(filename)
    print(minsup)

    res_dt = [np.mean(i) for i in res_dt]
    res_sr = [np.mean(i) for i in res_sr]
    res_rf = [np.mean(i) for i in res_rf]
    res_svc = [np.mean(i) for i in res_svc]
    res_knn = [np.mean(i) for i in res_knn]

    plt.figure()
    plt.plot(minsup, res_dt, marker='s', markersize=8, label="Decision tree", color="C1", alpha=0.7)
    plt.plot(minsup, res_sr, marker='^', markersize=7, label="Sequential rule", color="C0", alpha=0.7)
    plt.plot(minsup, res_rf, marker='o', markersize=7, label="Random forest", color="C2", alpha=0.7)
    plt.plot(minsup, res_svc, marker='x', markersize=7, label="SVC", color="C3", alpha=0.7)
    plt.plot(minsup, res_knn, marker='.', markersize=7, label="kNN", color="C4", alpha=0.7)
    plt.legend()
    plt.xlabel("Value of the minimum support (minsup)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy with the evolution of the minimum support")
    plt.ylim(top=0.8, bottom=0.5)
    plt.grid()
    plt.savefig("accuracy_minsup_big_30.svg")
    plt.show()


def analyze_best_params(pos='data/molecules-medium.pos', neg='data/molecules-medium.neg'):
    x = list(range(40, 200, 10))
    y = list(range(85, 101, 5))
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)

    with open('best_params_RF_2.txt', "a+") as fd:
        fd.write("{} {} {}\n".format(40, 200, 10))
        fd.write("{} {} {}\n".format(65, 81, 5))
        fd.flush()

        ii = range(len(x))
        jj = range(len(y))
        for minsup in tqdm(ii):
            for k in jj:
                out = np.mean(another_classifier(pos, neg, y[k], x[minsup], 'rf'))
                Z[k, minsup] = out
                fd.write('{} {} {}\n'.format(minsup, 16 + k, out))
                fd.flush()


def graph_best_params(filename):
    with open(filename, "r") as fd:

        x = np.array(list(range(40, 200, 10)))
        y = np.array(list(range(1, 101, 5)))
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(Y, dtype=float)

        for line in fd:
            i, j, a = line.split(" ")
            Z[int(j), int(i)] = float(a)

        plt.pcolor(X, Y, Z)
        plt.colorbar()
        plt.xlabel("minsup")
        plt.ylabel("k")
        plt.title("Heat map of the accuracy of the RF on the medium dataset")
        plt.savefig('heatmap_rf_medium.svg')
        plt.show()


def repare_file(filename):
    with open(filename, "r") as fd:
        for l in fd:
            tab = l.split(" ")
            with open(filename[:-4]+"_OK.txt", "a+") as fd2:
                i = 0
                j = 0
                index = 0
                for ee in range(len(tab)//2):
                    current_j = int(tab[index+1])
                    val = tab[index+2]
                    if current_j < j:
                        i += 1
                    j = current_j
                    fd2.write("{} {} {}\n".format(i, j, val))
                    index += 2


if __name__ == "__main__":
    # analyze_by_k()
    # analyze_by_minsup()
    # second = analyze_by_k()
    # analyze_by_minsup()
    # graph_from_minsup("accuracy_minsup_medium_30_balanced.txt", 7)
    # graph_from_minsup("accuracy_minsup_medium.npy", 7)
    # graph_from_minsup("accuracy_minsup_big_30_balanced.txt", 7)
    # graph_from_k("accuracy_k_medium_50_balanced.txt")
    # graph_from_k("accuracy_k_big_1200_balanced.txt")
    # analyze_best_params()
    # analyze_by_k()
    # analyze_by_minsup()
    graph_best_params("best_params_RF.txt")
