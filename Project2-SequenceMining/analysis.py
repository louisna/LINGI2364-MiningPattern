import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        wacc_res[i] = wacc.performance(pos_file, neg_file, k[i])
        info_res[i] = info.performance(pos_file, neg_file, k[i])
        abs_wracc_res[i] = abs_wracc.performance(pos_file, neg_file, k[i])

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


def zone_scoring_function_wacc(k=15, pos_file="Datasets/Protein/SRC1521.txt", neg_file="Datasets/Protein/PKA_group15.txt"):

    a = wacc.zone_analysis(pos_file, neg_file, k)

    o_x = []
    o_y = []
    max_x = 0
    max_y = 0
    for s, _ in a.best_k:
        for ss, pos, neg in s:
            o_x.append(neg)
            o_y.append(pos)
            max_x = max(max_x, neg)
            max_y = max(max_y, pos)
    P = a.P
    N = a.N
    score = a.min_Wacc

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Compute the limit of the scoring function
    x = list(range(max_x + round(max_x*0.1)))
    y1 = [(score + i/N) * P for i in x]
    y2 = [(-score + i/N) * P for i in x]
    # Plot the scoring function
    plt.plot(x, y1, color="C1", label="Scoring function")
    plt.plot(x, y2, color="C1")
    # Fill area above the scoring function
    plt.fill_between(x, y1, max(y1), color='C1', alpha=0.3)
    # Fill area below the scoring function
    plt.fill_between(x, y2, color='C1', alpha=0.3)

    # Compute the height and the width of the pruning zone
    height = score * P
    width = score * N

    # Plot the rectangle of the pruning zone
    rect = plt.Rectangle((0, 0), width, height, color="C2", alpha=0.3)
    ax.add_patch(rect)
    plt.plot([0, width], [height, height], color="C2", label="Pruning zone")
    plt.plot([width, width], [0, height], color="C2")

    # Plot the elements
    plt.scatter(o_x, o_y, marker='.', label="Top-k sequences")

    plt.ylim(bottom=0, top=max_y+5)
    plt.xlim(left=0, right=max_x+5)

    plt.legend()
    plt.xlabel("N space")
    plt.ylabel("P space")
    plt.title("ROC analysis for the Wacc scoring function and k=" + str(k))
    plt.savefig("ROC_wacc.svg")

    plt.show()


def zone_scoring_function_wracc(k=15, pos_file="Datasets/Protein/SRC1521.txt", neg_file="Datasets/Protein/PKA_group15.txt"):
    a = wracc.zone_analysis(pos_file, neg_file, k)

    o_x = []
    o_y = []
    max_x = 0
    max_y = 0
    for s, _ in a.best_k:
        for ss, pos, neg in s:
            o_x.append(neg)
            o_y.append(pos)
            max_x = max(max_x, neg)
            max_y = max(max_y, pos)
    P = a.P
    N = a.N
    score = a.min_Wacc

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # TODO
    # Compute the limit of the scoring function
    x = list(range(max_x + round(max_x*0.1)))
    y1 = [(score + i/N) * P for i in x]
    y2 = [(-score + i/N) * P for i in x]
    # Plot the scoring function
    plt.plot(x, y1, color="C1", label="Scoring function")
    plt.plot(x, y2, color="C1")
    # Fill area above the scoring function
    plt.fill_between(x, y1, max(y1), color='C1', alpha=0.3)
    # Fill area below the scoring function
    plt.fill_between(x, y2, color='C1', alpha=0.3)

    # TODO
    # Compute the height and the width of the pruning zone
    height = score * P
    width = score * N

    # Plot the rectangle of the pruning zone
    rect = plt.Rectangle((0, 0), width, height, color="C2", alpha=0.3)
    ax.add_patch(rect)
    plt.plot([0, width], [height, height], color="C2", label="Pruning zone")
    plt.plot([width, width], [0, height], color="C2")

    # Plot the elements
    plt.scatter(o_x, o_y, marker='.', label="Top-k sequences")

    plt.ylim(bottom=0, top=max_y+5)
    plt.xlim(left=0, right=max_x+5)

    plt.legend()
    plt.xlabel("N space")
    plt.ylabel("P space")
    plt.title("ROC analysis for the Wacc scoring function and k=" + str(k))
    plt.savefig("ROC_wacc.svg")

    plt.show()


def zone_scoring_function_info_gain(k=15, pos_file="Datasets/Protein/SRC1521.txt", neg_file="Datasets/Protein/PKA_group15.txt"):
    a = info.zone_analysis(pos_file, neg_file, k)

    o_x = []
    o_y = []
    max_x = 0
    max_y = 0
    for s, _ in a.best_k:
        for ss, pos, neg in s:
            o_x.append(neg)
            o_y.append(pos)
            max_x = max(max_x, neg)
            max_y = max(max_y, pos)
    P = a.P
    N = a.N
    score = a.min_Wacc

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # TODO
    # Compute the limit of the scoring function
    x = list(range(max_x + round(max_x*0.1)))
    y1 = [(score + i/N) * P for i in x]
    y2 = [(-score + i/N) * P for i in x]
    # Plot the scoring function
    plt.plot(x, y1, color="C1", label="Scoring function")
    plt.plot(x, y2, color="C1")
    # Fill area above the scoring function
    plt.fill_between(x, y1, max(y1), color='C1', alpha=0.3)
    # Fill area below the scoring function
    plt.fill_between(x, y2, color='C1', alpha=0.3)

    # TODO
    # Compute the height and the width of the pruning zone
    height = score * P
    width = score * N

    # Plot the rectangle of the pruning zone
    rect = plt.Rectangle((0, 0), width, height, color="C2", alpha=0.3)
    ax.add_patch(rect)
    plt.plot([0, width], [height, height], color="C2", label="Pruning zone")
    plt.plot([width, width], [0, height], color="C2")

    # Plot the elements
    plt.scatter(o_x, o_y, marker='.', label="Top-k sequences")

    plt.ylim(bottom=0, top=max_y+5)
    plt.xlim(left=0, right=max_x+5)

    plt.legend()
    plt.xlabel("N space")
    plt.ylabel("P space")
    plt.title("ROC analysis for the Wacc scoring function and k=" + str(k))
    plt.savefig("ROC_wacc.svg")

    plt.show()


def zone_scoring_function_absolute_wracc(k=15, pos_file="Datasets/Protein/SRC1521.txt", neg_file="Datasets/Protein/PKA_group15.txt"):
    a = wacc.zone_analysis(pos_file, neg_file, k)

    o_x = []
    o_y = []
    max_x = 0
    max_y = 0
    for s, _ in a.best_k:
        for ss, pos, neg in s:
            o_x.append(neg)
            o_y.append(pos)
            max_x = max(max_x, neg)
            max_y = max(max_y, pos)
    P = a.P
    N = a.N
    score = a.min_Wacc

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # TODO
    # Compute the limit of the scoring function
    x = list(range(max_x + round(max_x*0.1)))
    y1 = [(score + i/N) * P for i in x]
    y2 = [(-score + i/N) * P for i in x]
    # Plot the scoring function
    plt.plot(x, y1, color="C1", label="Scoring function")
    plt.plot(x, y2, color="C1")
    # Fill area above the scoring function
    plt.fill_between(x, y1, max(y1), color='C1', alpha=0.3)
    # Fill area below the scoring function
    plt.fill_between(x, y2, color='C1', alpha=0.3)

    # TODO
    # Compute the height and the width of the pruning zone
    height = score * P
    width = score * N

    # Plot the rectangle of the pruning zone
    rect = plt.Rectangle((0, 0), width, height, color="C2", alpha=0.3)
    ax.add_patch(rect)
    plt.plot([0, width], [height, height], color="C2", label="Pruning zone")
    plt.plot([width, width], [0, height], color="C2")

    # Plot the elements
    plt.scatter(o_x, o_y, marker='.', label="Top-k sequences")

    plt.ylim(bottom=0, top=max_y+5)
    plt.xlim(left=0, right=max_x+5)

    plt.legend()
    plt.xlabel("N space")
    plt.ylabel("P space")
    plt.title("ROC analysis for the Wacc scoring function and k=" + str(k))
    plt.savefig("ROC_wacc.svg")

    plt.show()


if __name__ == "__main__":
    # execution_time_analysis(max_k=21)
    zone_scoring_function_wacc()
