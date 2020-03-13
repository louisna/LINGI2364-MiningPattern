"""
Skeleton file for the project 1 of the LINGI2364 course.
Use this as your submission file. Every piece of code that is used in your program should be put inside this file.

This file given to you as a skeleton for your implementation of the Apriori and Depth
First Search algorithms. You are not obligated to use them and are free to write any class or method as long as the
following requirements are respected:

Your apriori and alternativeMiner methods must take as parameters a string corresponding to the path to a valid
dataset file and a double corresponding to the minimum frequency.
You must write on the standard output (use the print() method) all the itemsets that are frequent in the dataset file
according to the minimum frequency given. Each itemset has to be printed on one line following the format:
[<item 1>, <item 2>, ... <item k>] (<frequency>).
Tip: you can use Arrays.toString(int[] a) to print an itemset.

The items in an itemset must be printed in lexicographical order. However, the itemsets themselves can be printed in
any order.

Do not change the signature of the apriori and alternative_miner methods as they will be called by the test script.

__authors__ = GROUP 23, Edgar Gevorgyan (2018-16-00) AND Louis Navarre (1235-16-00)
"""

import itertools
import time

class Dataset:
	"""Utility class to manage a dataset stored in a external file."""

	def __init__(self, filepath):
		"""reads the dataset file and initializes files"""
		#self.transactions = list()
		self.items = set()
		self.dico = {}
		self.trans_number = 0

		try:
			lines = [line.strip() for line in open(filepath, "r")]
			lines = [line for line in lines if line]  # Skipping blank lines
			for line in lines:
				transaction = list(map(int, line.split(" ")))
				self.trans_number += 1
				for item in transaction:
					self.dico[tuple([item])] = self.dico.get(tuple([item]),[])
					self.dico[tuple([item])].append(transaction)
					self.items.add(item)
		except IOError as e:
			print("Unable to read dataset file!\n" + e)

	def trans_num(self):
		"""Returns the number of transactions in the dataset"""
		return self.trans_number

	def items_num(self):
		"""Returns the number of different items in the dataset"""
		return len(self.items)
	"""
	def get_transaction(self, i):
		##Returns the transaction at index i as an int array
		return self.transactions[i]
	"""


def is_subset(subset, settt):
	"""
	Determines if 'subset' is a subset of 'set'
	:param subset: The subset
	:param set: The set
	:return: True if 'subset' is a subset of 'set, False otherwise
	"""
	cpos = 0
	for c in settt:
		""" too slow
		if c > subset[cpos]:
			return False
		"""
		cpos += c == subset[cpos]  # Found the char
		if cpos == len(subset):
			return True
	return False


def is_prefix(s1, s2):
	"""
	Checks if s1 and s2 have the same prefix. Two itemsets have the same prefix if they are the same
	except for the last item. Both itemsets should be sorted following the alphabetical order
	:param s1: First itemset
	:param s2: Second itemset
	:return: True if s1 and s2 have the same prefix
	"""
	return s1[:-1] == s2[:-1]


def gen_supersets_prefix(sets):
	"""
	Generates supersets from 'sets'. Two itemsets can generate a unique superset if they have the same
	prefix (see function is_prefix). Consider s1 and s2 having the same prefix. The superset of s1 and s2 will be
	s1 + s2[last_item]. All itemsets should be sorted following the alphabetical order
	:param sets: The sets that will generate the supersets
	:return: The superset generated from 'sets'
	"""
	ret = []
	for i, s1 in enumerate(sets):
		for s2 in sets[i+1:]:
			if is_prefix(s1, s2):
				b = s1.copy()
				b.append(s2[-1])
				ret.append(b)
			else:
				break
	return ret


def apriori(filepath, minFrequency):
	"""Runs the apriori algorithm on the specified file with the given minimum frequency"""

	# Dataset object
	ds = Dataset(filepath)
	dico = ds.dico

	# Current sets working
	working_set = [[i] for i in ds.items]

	for level in range(ds.items_num()):  # Check each level
		# Frequent set
		frequent = []

		# Future dico
		future_dico = dict()

		# Monotonicity property
		if len(working_set) == 0:  # No more frequent set
			break
		for subset in working_set:
			freq_trans = []
			support = 0
			if level == 0:
				# Fast checking for the first level: only consier the size of the vertical representation of the DB
				support = len(dico[tuple(subset)]) 
			else:
				trans = dico[tuple(subset[:-1])]
				for i, seti in enumerate(trans):
					# If the remaining number of transactions is lower than the required frequency
					if len(trans) - i < minFrequency * ds.trans_num() - support:
						break  # Useless to continue
					if is_subset(subset, seti):
						support += 1
						freq_trans.append(seti)
			frequency = support / ds.trans_num()
			if frequency >= minFrequency:  # If item frequent
				frequent.append(subset)
				if level > 0:
					# Add the transactions containing this frequent itemset to the dictionary for the next level
					# Kind of a vertical representation
					future_dico[tuple(subset)] = freq_trans
				else:
					# Particular case for the level 0
					future_dico[tuple(subset)] = dico[tuple(subset)]
				print(list(subset), "({})".format(frequency))
		# Update the dico
		dico = future_dico
		# Generate supersets of the next level
		working_set = gen_supersets_prefix(frequent)


def alternative_miner(filepath, minFrequency):
	"""Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
	# Dataset object
	ds = Dataset(filepath)
	dico = ds.dico

	working_set = [[i] for i in ds.items]

	for i, itemset in enumerate(working_set):
		dfs(itemset, dico, ds, minFrequency, i, working_set)


def dfs(itemset, dico, ds, minFrequency, i, working_set):
	"""
	Simple DFS search at the node itemset
	:param itemset: the itemset
	:param dico: dictionary containing the transactions
	:param ds: the dataset
	:param minFrequency: the minimum frequency for the items
	:param i: ?/? c'est quoi
	:param working_set: the working set
	:return: void method
	"""
	if is_frequent_individual(itemset, dico, ds, minFrequency):
		# Visit the node
		freq = visit(itemset, dico, ds, minFrequency)
		# If frequent itemset and ?/? la condition signifie quoi ?
		if freq and i+1 < len(working_set):
			# ?/? working_set[i+1:] contient quoi ?
			for j, e in enumerate(working_set[i+1:]):
				a = itemset.copy()
				a.append(e[0])
				# Recursive call
				dfs(a, dico, ds, minFrequency, i+1+j, working_set)


def is_frequent_individual(itemset, dico, ds, minFrequency):
	"""
	Checks if the itemset contains only frequent items
	:param itemset: the itemset
	:param dico: dictionary containing the transactions
	:param ds: dataset
	:param minFrequency: the minimum frequency for the items
	:return: True if all items in itemset are frequent, False otherwise
	"""
	for e in itemset:
		support = len(dico[tuple([e])]) 
		frequency = support / ds.trans_num()
		if frequency < minFrequency:
			return False
	return True


def visit(itemset, dico, ds, minFrequency):
	"""
	Visit the node itemset, and change the dictionary to add the frequent items of the itemset
	:param itemset: the itemset to be visited
	:param dico: the dictionnary containing projected databases
	:param ds: the dataset
	:param minFrequency: the minimum frequency to be considered frequent
	:return: True if the itemset is frequent, False otherwise
	"""
	is_frequent = False
	freq_trans = []
	support = 0
	if len(itemset) == 1:
		support = len(dico[tuple(itemset)]) 
	else:
		trans = dico[tuple(itemset[:-1])]
		for i, seti in enumerate(trans):
			if len(trans) - i < minFrequency * ds.trans_num() - support:
				break  # Useless to continue
			if is_subset(itemset, seti):
				support += 1
				freq_trans.append(seti)
	frequency = support / ds.trans_num()
	if frequency >= minFrequency:
		is_frequent = True
		if len(itemset) > 1:
			dico[tuple(itemset)] = freq_trans
		print(list(itemset), "({})".format(frequency))
	return is_frequent
