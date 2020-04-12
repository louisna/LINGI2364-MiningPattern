import sys
import random
random.seed(1998)

epsilon = 10 ** -5


class Datasets:
    def __init__(self, pos, neg, all_symbols, bestk):
        self.pos = pos
        self.neg = neg
        self.all_symbols = [i for i in all_symbols]
        self.bestk = bestk
        self.first_pruning()

    def first_pruning(self):
        best_unique = []
        for symbol in self.all_symbols:
            sup_pos = self.pos.vertical_first.get((symbol,)[0], [])
            sup_neg = self.neg.vertical_first.get((symbol,)[0], [])
            wracc = Wracc(self.bestk.P, self.bestk.N, len(sup_pos), len(sup_neg))
            min_wracc = sys.maxsize
            in_list = False
            if len(best_unique) < self.bestk.k:
                best_unique.append(wracc)
            else:
                for s in best_unique:
                    if s == wracc:
                        in_list = True
                    min_wracc = min(min_wracc, s)
                if not in_list and min_wracc < wracc:
                    best_unique.remove(min_wracc)
                    best_unique.append(wracc)

        new_pos = dict()
        new_neg = dict()

        min_wracc = min(best_unique)

        new_all_symbols = []

        for symbol in self.all_symbols:
            sup_pos = self.pos.vertical_first.get((symbol,)[0], [])
            sup_neg = self.neg.vertical_first.get((symbol,)[0], [])
            coef = ((self.bestk.P / (self.bestk.P + self.bestk.N)) * (self.bestk.N / (self.bestk.P + self.bestk.N)))
            threshold = (min_wracc / coef) * self.bestk.P
            threshold2 = (min_wracc / coef) * self.bestk.N
            if Wracc(self.bestk.P, self.bestk.N, len(sup_pos), len(sup_neg)) >= min_wracc or len(
                    sup_pos) >= threshold or len(sup_neg) >= threshold2:
                new_all_symbols.append(symbol)

                # Remove
                # del self.pos.vertical[(symbol,)[0]]
                # del self.neg.vertical[(symbol,)[0]]
                if symbol in self.pos.symbols:
                    new_pos[(symbol,)[0]] = self.pos.vertical[(symbol,)[0]]
                if symbol in self.neg.symbols:
                    new_neg[(symbol,)[0]] = self.neg.vertical[(symbol,)[0]]
        self.pos.vertical = new_pos
        self.neg.vertical = new_neg
        if len(best_unique) == self.bestk.k:
            self.bestk.min_Wracc = min_wracc
        self.all_symbols = new_all_symbols

    def post_pruning_closed(self):
        # This is a decorator (private joke)
        def sublist(lst1, lst2):
            index = 0
            for i in lst2:
                if i == lst1[index]:
                    index += 1
                if index == len(lst1):
                    return True
            return False

        new_bestk = []
        for that_support, support in self.bestk.best_k:
            new_that_sup = []
            for sequence, sup_pos, sup_neg in that_support:
                is_sublist = False
                for sequence2, sup_pos2, sup_neg2 in that_support:
                    if len(sequence) < len(sequence2) and sublist(sequence,
                                                                  sequence2) and sup_pos == sup_pos2 and sup_neg == sup_neg2:  # Add sequence
                        is_sublist = True
                        break
                if not is_sublist:
                    new_that_sup.append((sequence, sup_pos, sup_neg))
            new_bestk.append((new_that_sup, support))
        self.bestk.best_k = new_bestk



class Dataset:
    """Utility class to manage a dataset stored in a external file."""

    def __init__(self, filepath):
        """
        reads the dataset file and initializes files
        Stores in the object a vertical representation of the dataset:
        map of symbols. For each symbol, a tuple of the transaction number + the position of the occurrence of this
        symbol.
        PAY ATTENTION ! Transactions number AND positions start at index 0 ! Not the same as in the Datasets.
        """
        self.transactions = list()
        self.symbols = set()
        self.vertical = dict()
        self.nb_trans = 0
        self.symbols_list = []
        self.vertical_first = dict()
        self.open_vertical(filepath)

        # print(self.vertical)

    def open_vertical(self, filepath):
        with open(filepath, "r") as fd:
            tr_nb = -1
            for line in fd:
                if not line or line == "\n":
                    tr_nb += 1
                    continue
                symbol, place = list(line.split(" "))
                self.symbols.add(symbol)
                a = tuple()
                a += (symbol,)
                self.vertical[a[0]] = self.vertical.get(a[0], [])
                self.vertical[a[0]].append((tr_nb, int(place)-1))

                new_vertical_first = self.vertical_first.get(a[0], [])
                if new_vertical_first:
                    if new_vertical_first[-1] != tr_nb:
                        new_vertical_first.append(tr_nb)
                else:
                    new_vertical_first.append(tr_nb)
                self.vertical_first[a[0]] = new_vertical_first

        self.nb_trans += tr_nb
        for i in self.symbols:
            self.symbols_list.append(i)

    def open_file_naive(self, filepath):
        with open(filepath, "r") as fd:
            tr = list()
            for line in fd:
                if not line or line == "\n":
                    if not len(tr) == 0:
                        self.transactions.append(tr)
                    tr = list()
                else:
                    symbol = list(line.split(" "))
                    tr.append(symbol[0])
                    self.symbols.add(symbol[0])

    def trans_num(self):
        """Returns the number of transactions in the dataset"""
        return self.nb_trans

    def items_num(self):
        """Returns the number of different items in the dataset"""
        return len(self.symbols)

    def get_transaction(self, i):
        """Returns the transaction at index i as an int array"""
        return self.transactions[i]


class BestK:
    """
    Each element inside best_k is tuple:
    - a list containing tuples with the same total support:
        * sequence
        * support in positive
        * support in negative
    - total support for this type
    """
    def __init__(self, k, P, N):
        self.k = k
        self.best_k = []
        self.min_Wracc = 0
        self.P = P
        self.N = N

    def add_frequent(self, sequence, support_pos, support_neg):
        wacc = Wracc(self.P, self.N, support_pos, support_neg)
        if wacc < self.min_Wracc:
            return
        if len(self.best_k) < self.k:
            # Not full
            for i in range(len(self.best_k)):
                (sequences, support) = self.best_k[i]
                if support == wacc:
                    # Already an existing
                    # TODO: Check if subsequence with same support and remove it
                    sequences.append((sequence, support_pos, support_neg))
                    # self.closing(sequence, support_pos, support_neg, sequences, support)
                    return
            # Not in there
            self.best_k.append(([(sequence, support_pos, support_neg)], wacc))
            self.best_k.sort(key=lambda b: b[1])
        else:  # Full

            # Check if this min support is already there
            for i in range(len(self.best_k)):
                (sequences, support) = self.best_k[i]
                if abs(support - wacc) < epsilon:
                    # Already an existing
                    # TODO: Check if subsequence with same support and remove it
                    sequences.append((sequence, support_pos, support_neg))
                    # self.closing(sequence, support_pos, support_neg, sequences, support)
                    return
            # Not in there => remove first
            self.best_k.pop(0)
            self.best_k.append(([(sequence, support_pos, support_neg)], wacc))
            self.best_k.sort(key=lambda b: b[1])

            self.min_Wracc = self.best_k[0][1]

    def print_bestk(self):
        for that_support, support in self.best_k:
            for sequence, sup_pos, sup_neg in that_support:
                st = ""
                for elem in sequence:
                    st = st + elem + ", "
                st = st[:-2]
                print('[{}]'.format(st), sup_pos, sup_neg, support)
                pass

    def closing(self, sequence, support_pos, support_neg, sequences, support):
        # Assume same frequence, just have to check if subsequence and remove it
        new_sequences = sequences.copy()
        # sequence, support positive, support negative
        for (s, sp, sn) in new_sequences:
            # print(s, sequence[:-1], sequence[1:] == s or sequence[:-1] == s)
            if len(s) + 1 == len(sequence):  # Could be ? Never Know
                if all(i in sequence for i in s):  # s is subsequence
                    # print("Nous faisons un remove de", s, "par", sequence)
                    sequences.remove((s, sp, sn))
            elif len(sequence) < len(s):  # Add subset ?
                if all(i in s for i in sequence):
                    return  # Not append
        sequences.append((sequence, support_pos, support_neg))


def Wracc(P, N, p, n):
    return abs(round(((P/(P+N))*(N/(P+N)))*(p/P - n/N), 5))


def projection(added_symbol, bestK, proj):
    # Contains the projected database (positive) for the new sequence containing symbol
    new_proj = dict()
    first_occ_added_symbol = first_occ(added_symbol, proj)
    if not first_occ_added_symbol:
        return new_proj
    # Project the positive database
    for symbol, trans in proj.items():
        # The new array containing the projection for 'symbol' with the addition of added_symbol
        new_freq_trans = filter_trans(trans, first_occ_added_symbol)
        # if len(new_freq_trans) >= bestK.min_total_support:
        a = tuple()
        a += (symbol,)
        new_proj[a[0]] = new_freq_trans
    return new_proj


def first_occ(added_symbol, proj):
    first = []

    a = tuple()
    a += (added_symbol,)
    processed = proj.get(a[0], [])
    if not processed:
        return first
    last_trans_index = processed[0][0]
    first.append(processed[0])
    for i in processed:
        if i[0] != last_trans_index:
            first.append(i)
            last_trans_index = i[0]
    return first


def filter_trans(trans, first_occ_added_symbol):
    # TODO: SPEED-UP: check if the remaining transactions are not enough to make it frequent item
    out = []
    index_first = 0
    # tn = transaction number, tp = transaction position
    for (tn, tp) in trans:
        while first_occ_added_symbol[index_first][0] < tn:
            index_first += 1
            if index_first >= len(first_occ_added_symbol):
                return out  # Visited all
        if tn < first_occ_added_symbol[index_first][0]:
            continue  # Current tn not in the projection
        # Assume same transaction number now
        # Position in the transaction is bigger
        if tp > first_occ_added_symbol[index_first][1]:
            out.append((tn, tp))
    return out


def count_occurences_symbol(projection, symbol):
    return len(first_occ(symbol, projection))


def SPADE(data_pos, data_neg, bestk):

    all_symbols = data_pos.symbols.union(data_neg.symbols)
    all_symbols_list = [i for i in all_symbols]
    all_symbols_list.sort()

    glob = Datasets(data_pos, data_neg, all_symbols, bestk)

    dfs([], bestk, glob, data_pos.vertical, data_neg.vertical)

    # Print the result
    glob.post_pruning_closed()
    bestk.print_bestk()


def dfs(sequence, bestk, dss, proj_pos, proj_neg):
    # if sequence == ['C']:
    #     print("passage DEBUT")
    #     print(bestk.min_total_support)
    #     print(proj_pos)
    for symbol in dss.all_symbols:
        a = tuple()
        a += (symbol,)
        if a[0] in proj_pos.keys() or a[0] in proj_neg.keys():
            # Support before projection
            support_pos = count_occurences_symbol(proj_pos, symbol)
            support_neg = count_occurences_symbol(proj_neg, symbol)
            coef = ((bestk.P / (bestk.P + bestk.N)) * (bestk.N / (bestk.P + bestk.N)))
            threshold_pos = (bestk.min_Wracc / coef) * bestk.P
            threshold_neg = (bestk.min_Wracc / coef) * bestk.N
            # Threshold set to 0
            wracc = Wracc(bestk.P, bestk.N, support_pos,
                     support_neg)
            if wracc >= bestk.min_Wracc or support_pos >= threshold_pos or support_neg >= threshold_neg:
                # Frequent symbol in this sequence
                new_pos = projection(symbol, bestk, proj_pos)
                new_neg = projection(symbol, bestk, proj_neg)

                if len(new_pos) + len(new_neg) > 0:
                    new_sequence = sequence.copy()
                    new_sequence.append(symbol)

                    # if sequence == ['C']:
                    #    print("je suis la")

                    bestk.add_frequent(new_sequence, support_pos, support_neg)
                    dfs(new_sequence, bestk, dss, new_pos, new_neg)
    # if sequence == ["C"]:
    #    print("passage FIN")


def main():
    pos_filepath = sys.argv[1]  # filepath to positive class file
    neg_filepath = sys.argv[2]  # filepath to negative class file
    k = int(sys.argv[3])

    data_pos = Dataset(pos_filepath)
    data_neg = Dataset(neg_filepath)
    bestk = BestK(k, data_pos.trans_num(), data_neg.trans_num())

    SPADE(data_pos, data_neg, bestk)


if __name__ == "__main__":
    main()


# python3 supervised_closed_sequence_mining_absolute_wracc.py Datasets/Protein/PKA_group15.txt Datasets/Protein/SRC1521.txt 6
# python3 supervised_closed_sequence_mining_absolute_wracc.py Datasets/Test/positive.txt Datasets/Test/negative.txt 6
# python3 supervised_closed_sequence_mining_absolute_wracc.py Datasets/Reuters/earn.txt Datasets/Reuters/acq.txt 1
