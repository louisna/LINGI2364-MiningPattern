import numpy as np
import time
import matplotlib.pyplot as plt

import supervised_closed_sequence_mining as wracc
import supervised_closed_sequence_mining_absolute_wracc as abs_wracc
import supervised_closed_sequence_mining_absolute_wacc as wacc
import supervised_closed_sequence_mining_info_gain as info


def execution_time_analysis(max_k=11, pos_file="Datasets/Protein/SRC1521.txt", neg_file="Datasets/Protein/PKA_group15.txt"):
    k = [i for i in range(1, max_k)]
    wracc_res = [0.0] * len(k)
    abs_wracc_res = [0.0] * len(k)
    wacc_res = [0.0] * len(k)
    info_res = [0.0] * len(k)

    for i in range(len(k)):
        print(i)
        wracc_res[i] = wracc.performance(pos_file, neg_file, k[i])
        abs_wracc_res[i] = abs_wracc.performance(pos_file, neg_file, k[i])
        wacc_res[i] = wacc.performance(pos_file, neg_file, k[i])
        info_res[i] = info.performance(pos_file, neg_file, k[i])

    # plt.plot(freq[:l1], a1, marker='o', markersize=7, label="Apriori", linestyle='dashed', alpha=0.8, color="C0")
    plt.plot(k, wracc_res, marker='s', markersize=7, label="Wracc", color="C0")
    plt.plot(k, abs_wracc_res, marker='^', markersize=7, label="Absolute Wracc", color="C1")
    plt.plot(k, wacc_res, marker='o', markersize=7, label="Wacc", color="C2")
    plt.plot(k, info_res, marker='X', markersize=7, label="Information Gain", color="C3")
    plt.legend()
    plt.xlabel("Number of top frequent sequences")
    plt.ylabel("Execution time [s]")
    plt.title("Closed supervised sequence mining execution time")
    plt.grid()
    plt.savefig("execution_time_analysis.svg")


if __name__ == "__main__":
    execution_time_analysis(max_k=21)