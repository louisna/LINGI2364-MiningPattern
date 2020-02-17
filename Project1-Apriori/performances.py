from frequent_itemset_miner import *
import numpy as np
import time
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    plot_apriori_vs_dfs("./Datasets/chess.dat", 0.88, 0.98, 0.02)
