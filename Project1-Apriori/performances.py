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

perfLevel = [[
    3.51579213142395
],[
    3.4888179302215576
],[
    3.424783945083618
],[
    3.396592140197754,
    6.1725969314575195,
    7.429931163787842,
    7.9136199951171875,
    7.961974143981934
],[
    3.6056079864501953,
    14.33826994895935,
    21.433209896087646,
    28.541249990463257,
    32.365742683410645,
    33.44318389892578,
    33.585800886154175
],[
    3.459404945373535,
    42.719465017318726,
    81.00412011146545,
    123.79487299919128,
    151.6620578765869,
    161.90154814720154,
    163.88968300819397,
    164.05434226989746
]]

if __name__ == "__main__":
    #plot_apriori_vs_dfs("./Datasets/chess.dat", 0.88, 0.98, 0.02)
    print(perfLevel)