import sys
import random
random.seed(1998)

class Datasets:
    def __init__(self, pos, neg, all_symbols):
        self.pos = pos
        self.neg = neg
        self.all_symbols = all_symbols

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
                self.vertical[tuple([symbol])] = self.vertical.get(tuple([symbol]), [])
                self.vertical[tuple([symbol])].append((tr_nb, int(place)-1))

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
    def __init__(self, k):
        self.k = k
        self.best_k = []
        self.min_total_support = 1

    def add_frequent(self, sequence, support_pos, support_neg):
        if len(self.best_k) < self.k:
            # Not full
            for i in range(len(self.best_k)):
                (sequences, support) = self.best_k[i]
                if support == support_pos + support_neg:
                    # Already an existing
                    sequences.append((sequence, support_pos, support_neg))
                    return
            # Not in there
            self.best_k.append(([(sequence, support_pos, support_neg)], support_pos + support_neg))
            self.best_k.sort(key=lambda b: b[1])
        else:  # Full

            # Check if this min support is already there
            for i in range(len(self.best_k)):
                (sequences, support) = self.best_k[i]
                if support == support_pos + support_neg:
                    # Already an existing
                    sequences.append((sequence, support_pos, support_neg))
                    return
            # Not in there => remove first
            self.best_k.pop(0)
            self.best_k.append(([(sequence, support_pos, support_neg)], support_pos + support_neg))
            self.best_k.sort(key=lambda b: b[1])

            self.min_total_support = self.best_k[0][1]

    def print_bestk(self):
        for that_support, support in self.best_k:
            for sequence, sup_pos, sup_neg in that_support:
                st = ""
                for elem in sequence:
                    st = st + elem + ", "
                st = st[:-2]
                print('[{}]'.format(st), sup_pos, sup_neg, sup_pos + sup_neg)
                pass


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
        if len(new_freq_trans) >= bestK.min_total_support:  # Frequent item
            new_proj[tuple(symbol)] = new_freq_trans
    return new_proj


def first_occ(added_symbol, proj):
    first = []

    processed = proj.get(tuple(added_symbol), [])
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


def count_occurences(projection):
    total = 0
    for symbol in projection.keys():
        a = first_occ(symbol, projection)
        total += len(a)
    return total


def count_occurences_symbol(projection, symbol):
    return len(first_occ(symbol, projection))


def SPADE(data_pos, data_neg, bestk):

    # TODO: Start the search and DFS
    all_symbols = data_pos.symbols.union(data_neg.symbols)
    all_symbols_list = [i for i in all_symbols]
    all_symbols_list.sort()
    #for symbol in all_symbols_list:
    dfs([], bestk, Datasets(data_pos, data_neg, all_symbols), data_pos.vertical, data_neg.vertical)

    # Print the result
    bestk.print_bestk()


# TODO: Maybe error if pos & neg don't have exactly the same symbols
# TODO: Maybe the 2 checks are useless
def dfs(sequence, bestk, dss, proj_pos, proj_neg):
    for symbol in dss.all_symbols:
        if tuple(symbol) in proj_pos.keys() or tuple(symbol) in proj_neg.keys():
            support_pos = count_occurences_symbol(proj_pos, symbol)
            support_neg = count_occurences_symbol(proj_neg, symbol)
            support = support_pos + support_neg
            if support > bestk.min_total_support:  # Frequent symbol in this sequence
                new_pos = projection(symbol, bestk, proj_pos)
                new_neg = projection(symbol, bestk, proj_neg)
                if len(new_pos) + len(new_neg) > 0:
                    new_sequence = sequence.copy()
                    new_sequence.append(symbol)

                    bestk.add_frequent(new_sequence, support_pos, support_neg)
                    dfs(new_sequence, bestk, dss, new_pos, new_neg)


def main():
    pos_filepath = sys.argv[1]  # filepath to positive class file
    neg_filepath = sys.argv[2]  # filepath to negative class file
    k = int(sys.argv[3])

    data_pos = Dataset(pos_filepath)
    data_neg = Dataset(neg_filepath)
    bestk = BestK(k)

    SPADE(data_pos, data_neg, bestk)


if __name__ == "__main__":
    main()


# python3 frequent_sequence_mining.py Datasets/Protein/PKA_group15.txt Datasets/Protein/SRC1521.txt 1
# python3 frequent_sequence_mining.py Datasets/Test/positive.txt Datasets/Test/negative.txt 1



