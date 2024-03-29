#from frequent_itemset_miner import *
import numpy as np
import time
import matplotlib.pyplot as plt

freq = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

apriori_times = [
    [0.00015, 0.0003039, 0.000299, 0.000313, 0.000315, 0.00029, 0.00030, 0.000417, 0.00043, 0.0005],
    [0.1562, 10.62816, 119.0594],
    [3.598],
    [0.2315, 0.448605, 0.645667, 0.85617, 1.37973, 2.956, 9.625, 38.2326, 380.483],
    [3.2125, 3.423277, 3.39468, 7.7294, 33.9895, 151.208],
    [5.1089],
    [1.61187, 1.36656, 1.4029, 1.504348, 1.6686, 1.34818, 1.637, 1.7046, 1.67787, 1.9154],
    [16.43, 34.8564, 202.47]
]

dfs_times = [
    [0.000149, 0.00015, 0.000149, 0.000307, 0.000381, 0.000406, 0.00031, 0.00046, 0.000654, 0.0004],
    [0.1507, 15.69, 180.251],
    [3.921],
    [0.283, 0.474, 0.74469, 0.7852, 1.60527, 3.74387, 15.5614, 60.8258, 704.2694],
    [3.537, 3.3774, 3.49497, 8.8604, 67.627, 431.66],
    [5.429],
    [1.6123, 1.5365, 1.5656, 1.6984, 1.619, 1.42319, 1.559, 1.5142, 1.545, 2.0890],
    [16.844, 39.0213, 223.828]
]

#mushroom
#pumsb_star TIME OUT for all freq
# with prefix gen TIME OUT à 10 min
apriori_naive = [
    [2.702, 3.161, 3.625, 3.948, 5.1236, 11.227, 37.3219, 162.427]
]



def plot_result():
    a1 = apriori_times[3]
    l1 = len(a1)

    d1 = dfs_times[3]
    print(d1)
    l2 = len(d1)

    n1 = apriori_naive[0]
    l3 = len(n1)

    # plt.plot(freq[:l3], n1, marker='s', markersize=7, label="Naive Apriori", linestyle='dashed', color="C2")
    plt.plot(freq[:l1], a1, marker='o', markersize=7, label="Apriori", linestyle='dashed', alpha=0.8, color="C0")
    plt.plot(freq[:l2], d1, marker='^', markersize=7, label="DFS-based", linestyle='dashed', alpha=0.6, color="C1")
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.title("pumsb_star dataset")
    # plt.title("mushroom dataset")
    plt.xlabel("Frequency")
    plt.ylabel("Execution time (s)")
    plt.grid()

    plt.savefig("pumsb_star.png")
    # plt.savefig("mushroom.png")
    plt.show()


perfLevel = [[
    3.51579213142395, 0., 0., 0., 0., 0., 0., 0.
],[
    3.4888179302215576, 0., 0., 0., 0., 0., 0., 0.
],[
    3.424783945083618, 0., 0., 0., 0., 0., 0., 0.
],[
    3.396592140197754,
    6.1725969314575195,
    7.429931163787842,
    7.9136199951171875,
    7.961974143981934, 0., 0., 0.
],[
    3.6056079864501953,
    14.33826994895935,
    21.433209896087646,
    28.541249990463257,
    32.365742683410645,
    33.44318389892578,
    33.585800886154175, 0.
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

def plot_level():
    x = [1., 0.9, 0.8, 0.7, 0.6, 0.5]
    legend = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6-10", "Level 11-20", "Level 21+"]
    for i in range(7, -1, -1):
        plt.bar(x, [perfLevel[j][i] for j in range(6)], label=legend[i], width=0.07)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()


memory_aprori = [111.267171, 111.155237, 111.150701, 111.166941, 121.298752, 182.197952]
memory_dfs = [111.266579, 111.155237, 111.151013, 111.166941, 138.706028, 250.875528]
#mushroom memory analyse
apriori_naive_memory = [
    [3.507835,3.505159,3.505159,3.505159,3.505159,3.505159,3.505159,3.505159]
]
#mushroom memory analyse
apriori_memory = [
    [5.041407,5.029299,5.029299,5.203988,5.884556,8.227916,15.87904,40.19978]
]
#mushroom memory analyse
dfs_memory = [
    [5.040815,5.029299,5.106828,5.543244,6.313276,9.96942,22.26676,72.970324]
]
def plot_memory():
    x = [1., 0.9, 0.8, 0.7, 0.6, 0.5]
    plt.plot(x, memory_aprori, marker='o', markersize=7, label="Apriori", linestyle='dashed')
    plt.plot(x, memory_dfs, marker='^', markersize=7, label="DFS-based", linestyle='dashed')
    plt.gca().invert_xaxis()
    plt.xlabel("Minimum frequency")
    plt.ylabel("Maximum memory consummed during the search (MB)")
    plt.title("pumsb_star")
    plt.legend()
    plt.grid()
    plt.savefig("memory.png")
    plt.show()


if __name__ == "__main__":
    # plot_level()
    plot_result()
    # plot_memory()
