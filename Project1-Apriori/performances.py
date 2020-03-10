from frequent_itemset_miner import *
import numpy as np
import time
import matplotlib.pyplot as plt

freq = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

apriori_times = [
    [0.00015, 0.0003039, 0.000299, 0.000313, 0.000315, 0.00029, 0.00030, 0.000417, 0.00043, 0.0005],
    [0.1562, 10.62816, 119.0594],
    [3.598],
    [0.2315, 0.448605, 0.645667, 0.85617, 1.37973, 2.956, 9.625, 38.2326],
    [3.2125, 3.423277, 3.39468, 7.7294, 33.9895, 151.208],
    [5.1089],
    [1.61187, 1.36656, 1.4029, 1.504348, 1.6686, 1.34818, 1.637, 1.7046, 1.67787, 1.9154],
    [16.43, 34.8564, 202.47]
]

dfs_times = [
    [0.000149, 0.00015, 0.000149, 0.000307, 0.000381, 0.000406, 0.00031, 0.00046, 0.000654, 0.0004],
    [0.1507, 15.69, 180.251],
    [3.921],
    [0.283, 0.474, 0.74469, 0.7852, 1.60527, 3.74387, 15.5614, 60.8258],
    [3.537, 3.3774, 3.49497, 8.8604, 67.627],
    [5.429],
    [1.6123, 1.5365, 1.5656, 1.6984, 1.619, 1.42319, 1.559, 1.5142, 1.545, 2.0890],
    [16.844, 39.0213, 223.828]
]


def plot_result():
    a1 = apriori_times[4]
    l1 = len(a1)

    d1 = dfs_times[4]
    l2 = len(d1)
    plt.plot(freq[:l1], a1, marker='s', markersize=7, label="Apriori", linestyle='dashed')
    plt.plot(freq[:l2], d1, marker='^', markersize=7, label="DFS-based", linestyle='dashed')
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.title("pumsp_star dataset")
    plt.xlabel("Frequency")
    plt.ylabel("Execution time [s]")

    # plt.savefig("pumsb_star.png")
    plt.show()

"""
def plot_apriori_vs_dfs(filename, freq_min=0.7, freq_max=0.9, step=0.1):
    frequences = np.arange(freq_min, freq_max, step)
    res_apriori = []
    res_dfs = []
    for i, freq in enumerate(frequences):
        ti = time.time()
        apriori(filename, freq)
        res_apriori.append(time.time()-ti)
        ti = time.time()
        alternative_miner(filename, freq)
        res_dfs.append(time.time()-ti)

    plt.plot(frequences, res_apriori)
    plt.plot(frequences, res_dfs)
    plt.show()
"""


if __name__ == "__main__":
    plot_result()
