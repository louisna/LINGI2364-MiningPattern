import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
#from matplotlib_venn import venn3, venn3_circles


import supervised_closed_sequence_mining as wracc
import supervised_closed_sequence_mining_absolute_wracc as abs_wracc
import supervised_closed_sequence_mining_absolute_wacc as wacc
import supervised_closed_sequence_mining_info_gain as info
import supervised_closed_sequence_mining_chi_square_correlation as chi
import math as m


def execution_time_analysis(max_k=11, pos_file="Datasets/Protein/SRC1521.txt", neg_file="Datasets/Protein/PKA_group15.txt"):
    k = [i for i in range(1, max_k)]
    wracc_res = [0.0] * len(k)
    abs_wracc_res = [0.0] * len(k)
    # wacc_res = [0.0] * len(k)
    info_res = [0.0] * len(k)

    for i in range(len(k)):
        print(i)
        wracc_res[i] = wracc.performance(pos_file, neg_file, k[i])
        # wacc_res[i] = wacc.performance(pos_file, neg_file, k[i])
        info_res[i] = info.performance(pos_file, neg_file, k[i])
        abs_wracc_res[i] = abs_wracc.performance(pos_file, neg_file, k[i])

    # plt.plot(freq[:l1], a1, marker='o', markersize=7, label="Apriori", linestyle='dashed', alpha=0.8, color="C0")
    plt.plot(k, wracc_res, marker='s', markersize=8, label="Wracc", color="C0")
    plt.plot(k, abs_wracc_res, marker='^', markersize=7, label="|Wracc|", color="C1", alpha=0.7)
    # plt.plot(k, wacc_res, marker='o', markersize=7, label="Wacc", color="C2")
    plt.plot(k, info_res, marker='o', markersize=7, label="Information Gain", color="C2", alpha=0.7)
    plt.legend()
    plt.xlabel("Number of top frequent sequences")
    plt.ylabel("Execution time [s]")
    plt.title("Closed supervised sequence mining execution time")
    plt.grid()
    plt.savefig("execution_time_analysis.svg")
    plt.show()


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
    plt.fill_between(x, y1, max_y+5, color='C1', alpha=0.3)
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
    score = a.min_Wracc

    # To be like th others
    max_x = 372-5

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Compute the limit of the scoring function
    x = list(range(max_x + round(max_x*0.1)))
    cst = (P/(N+P)) * (N/(P+N))
    y = [((score/cst) + (i/N))*P for i in x]
    # Plot the scoring function
    plt.plot(x, y, color="C1", label="Scoring function")
    # Fill area above the scoring function
    plt.fill_between(x, y, max_y+5, color='C1', alpha=0.3)
    # Fill area below the scoring function
    # plt.fill_between(x, y2, color='C1', alpha=0.3)

    # Compute the height and the width of the pruning zone
    height = score * P / cst
    width = max_x*2

    # Plot the rectangle of the pruning zone
    rect = plt.Rectangle((0, 0), width, height, color="C2", alpha=0.3)
    ax.add_patch(rect)
    plt.plot([0, width], [height, height], color="C2", label="Pruning zone")
    plt.plot([width, width], [0, height], color="C2")

    # Plot the elements
    plt.scatter(o_x, o_y, marker='.', label="Top-k sequences")

    plt.ylim(bottom=0, top=max_y+5)
    plt.xlim(left=0, right=max_x+5)

    plt.legend(loc="upper right")
    plt.xlabel("N space")
    plt.ylabel("P space")
    plt.title("ROC analysis for the Wracc scoring function and k=" + str(k))
    plt.savefig("ROC_wracc.svg")

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
    score = a.min_impurity

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # TODO
    # Compute the limit of the scoring function
    x = list(range(max_x + round(max_x*0.1)))
    y = list(range(max_y + round(max_y*0.1)))
    y1 = [0]*len(x)
    for i in x:
        for j in y:
            if info.Impurity(P,N,j,i) - score <= 10 ** -7 and j > i:
                y1[i] = j
        if y1[i] >= 309:
            y1[i] = y1[i-1] + 0.07 * m.log(i+3)
        if y1[i] == 0:
            y1[i] = 360
    y2 = [0]*len(x)
    for i in x:
        for j in y:
            if score - info.Impurity(P,N,j,i) <= 10 ** -7 and j < i:
                y2[i] = j
    minY = 0
    for i in x:
        if (y2[i] > 0):
            minY = i
            break
    for i in x:
        if i >= minY:
            y2[i] += 4
        if i > y1[0] and i < minY:
            y2[i] = 1.045**(i- y1[0])
    # Plot the scoring function
    plt.plot(x, y1, color="C1", label="Scoring function")
    plt.plot(x, y2, color="C1")
    # Fill area above the scoring function
    plt.fill_between(x, y1, max_y+5, color='C1', alpha=0.3)
    # Fill area below the scoring function
    plt.fill_between(x, y2, color='C1', alpha=0.3)

    # TODO
    # Compute the height and the width of the pruning zone
    height = y1[0]
    width = height

    # Plot the rectangle of the pruning zone
    rect = plt.Rectangle((0, 0), width, height, color="C2", alpha=0.3)
    ax.add_patch(rect)
    plt.plot([0, width], [height, height], color="C2", label="Pruning zone")
    plt.plot([width, width], [0, height], color="C2")

    # Plot the elements
    plt.scatter(o_x, o_y, marker='.', label="Top-k sequences")

    plt.ylim(bottom=0, top=max_y+5)
    plt.xlim(left=0, right=max_x+5)

    plt.legend(loc="upper right")
    plt.xlabel("N space")
    plt.ylabel("P space")
    plt.title("ROC analysis for the Information Gain scoring function and k=" + str(k))
    plt.savefig("ROC_info.svg")

    plt.show()

def zone_scoring_function_chi_square_correlation(k=15, pos_file="Datasets/Protein/SRC1521.txt", neg_file="Datasets/Protein/PKA_group15.txt"):
    a = chi.zone_analysis(pos_file, neg_file, k)

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
    score = a.min_Chi

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # TODO
    # Compute the limit of the scoring function
    x = list(range(max_x + round(max_x*0.1)))
    y = list(range(max_y + round(max_y*0.1)))
    y1 = [0]*len(x)
    for i in x:
        for j in y:
            if chi.Chi_square(P,N,j,i) - score <= 10 ** -8 and j > i:
                y1[i] = j
        if y1[i] == 0:
            y1[i] = 360
    y2 = [0]*len(x)
    for i in x:
        for j in y:
            if score - chi.Chi_square(P,N,j,i) <= 10 ** -8 and j < i:
                y2[i] = j
        if y2[i] > 300:
            y2[i] = y2[i-1] + 0.01*(i-300)**1.25
    # Plot the scoring function
    plt.plot(x, y1, color="C1", label="Scoring function")
    plt.plot(x, y2, color="C1")
    # Fill area above the scoring function
    plt.fill_between(x, y1, max_y+5, color='C1', alpha=0.3)
    # Fill area below the scoring function
    plt.fill_between(x, y2, color='C1', alpha=0.3)

    # TODO
    # Compute the height and the width of the pruning zone
    height = y1[0]
    width = 0
    for i in x:
        if (y2[i] > 0):
            width = i
            break

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
    plt.title("ROC analysis for the Information Gain scoring function and k=" + str(k))
    plt.savefig("ROC_info.svg")

    plt.show()

def zone_scoring_function_absolute_wracc(k=15, pos_file="Datasets/Protein/SRC1521.txt", neg_file="Datasets/Protein/PKA_group15.txt"):
    a = abs_wracc.zone_analysis(pos_file, neg_file, k)

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
    score = a.min_Wracc

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Compute the limit of the scoring function
    x = list(range(max_x + round(max_x*0.1)))
    cst = (P / (N + P)) * (N / (P + N))
    y1 = [((score / cst) + (i / N)) * P for i in x]
    y2 = [((-score / cst) + (i / N)) * P for i in x]
    print('--------')
    print(y1)
    # Plot the scoring function
    plt.plot(x, y1, color="C1", label="Scoring function")
    plt.plot(x, y2, color="C1")
    # Fill area above the scoring function
    plt.fill_between(x, y1, max_y+5, color='C1', alpha=0.3)
    # Fill area below the scoring function
    plt.fill_between(x, y2, color='C1', alpha=0.3)

    # Compute the height and the width of the pruning zone
    height = score * P / cst
    width = score * N / cst

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
    plt.title("ROC analysis for the Absolute Wracc scoring function and k=" + str(k))
    plt.savefig("ROC_abs_wracc.svg")

    plt.show()

def scoring_statistics(k=15, pos_file="Datasets/Protein/SRC1521.txt", neg_file="Datasets/Protein/PKA_group15.txt"):
    w = wracc.zone_analysis(pos_file, neg_file, k)
    w_stat = np.zeros(len(w.best_k))
    aw = abs_wracc.zone_analysis(pos_file, neg_file, k)
    aw_stat = np.zeros(len(aw.best_k))
    i = info.zone_analysis(pos_file, neg_file, k)
    i_stat = np.zeros(len(i.best_k))
    c = chi.zone_analysis(pos_file, neg_file, k)
    c_stat = np.zeros(len(c.best_k))    
    for j in range(len(w.best_k)):
        w_stat[j] = w.best_k[j][1]
    for j in range(len(aw.best_k)):
        aw_stat[j] = aw.best_k[j][1]
    for j in range(len(i.best_k)):
        i_stat[j] = i.best_k[j][1]
    for j in range(len(c.best_k)):
        c_stat[j] = c.best_k[j][1]

    print("wracc :","max =",np.max(w_stat), "min =", np.min(w_stat), "mean =" ,np.mean(w_stat))
    print("abs_wracc :","max =",np.max(aw_stat), "min =", np.min(aw_stat), "mean =" ,np.mean(aw_stat))
    print("info :","max =",np.max(i_stat), "min =", np.min(i_stat), "mean =" ,np.mean(i_stat))
    print("chi :","max =",np.max(c_stat), "min =", np.min(c_stat), "mean =" ,np.mean(c_stat))


def similarities_analysis(k=15, pos_file="Datasets/Protein/SRC1521.txt", neg_file="Datasets/Protein/PKA_group15.txt"):
    def set_from_best(best):
        s = set()
        for sequences, _ in best.best_k:
            for sequence, _, _ in sequences:
                string = ""
                for i in sequence:
                    string += i
                s.add(string)
        return s

    wracc_k = wracc.zone_analysis(pos_file, neg_file, k)
    wracc_set = set_from_best(wracc_k)

    abs_wracc_k = abs_wracc.zone_analysis(pos_file, neg_file, k)
    abs_wracc_set = set_from_best(abs_wracc_k)

    info_k = info.zone_analysis(pos_file, neg_file, k)
    info_set = set_from_best(info_k)
    v3 = venn3([abs_wracc_set, wracc_set, info_set], alpha=0.4, set_labels=('|Wracc|', 'Wracc', 'Information gain'))
    for text in v3.set_labels:
        text.set_fontsize(15)
    plt.title("k = " + str(k), fontsize=15)
    plt.savefig("venn_diagram_" + str(k) + "_" + pos_file[-9:-4] + ".svg")
    plt.show()


if __name__ == "__main__":
    # execution_time_analysis(max_k=30)
    # zone_scoring_function_wacc()
    # zone_scoring_function_wracc()
    # zone_scoring_function_absolute_wracc()
    zone_scoring_function_info_gain()
    # zone_scoring_function_chi_square_correlation()
    # similarities_analysis(k=5)
    # similarities_analysis(pos_file="Datasets/Test/positive.txt",  neg_file='Datasets/Test/negative.txt', k=3)
