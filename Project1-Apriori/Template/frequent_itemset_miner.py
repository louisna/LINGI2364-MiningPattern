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
		self.transactions = list()
		self.items = set()

		try:
			lines = [line.strip() for line in open(filepath, "r")]
			lines = [line for line in lines if line]  # Skipping blank lines
			for line in lines:
				transaction = list(map(int, line.split(" ")))
				self.transactions.append(transaction)
				for item in transaction:
					self.items.add(item)
		except IOError as e:
			print("Unable to read dataset file!\n" + e)

	def trans_num(self):
		"""Returns the number of transactions in the dataset"""
		return len(self.transactions)

	def items_num(self):
		"""Returns the number of different items in the dataset"""
		return len(self.items)

	def get_transaction(self, i):
		"""Returns the transaction at index i as an int array"""
		return self.transactions[i]

def is_subset(subset, set):
	"""
	Determines if 'subset' is a subset of 'set'
	:param subset: The subset
	:param set: The set
	:return: True if 'subset' is a subset of 'set, False otherwise
	"""
	cpos = 0
	for c in set:
		if c == subset[cpos]:
			cpos += 1  # Found the char
		if cpos == len(subset):
			return True
	return False

def gen_supersets_naive(ds, level):
	"""
	Naive generation of the supersets, generating all permutations of size level+2
	:param ds: The object representing the dataset
	:param level: The level (~ size) of the sets
	:return: A list of the supersets at the given level
	"""
	return list(itertools.permutations(ds.items, level + 2))

def is_prefix(s1, s2):
	s1 = list(s1)
	s1.pop()
	s2 = list(s2)
	s2.pop()
	s1.sort()
	s2.sort()
	return s1 == s2

def gen_supersets_prefix(ds, level, sets):
	ret = []
	for i, s1 in enumerate(sets):
		for j, s2 in enumerate(sets[i:]):
			if is_prefix(s1, s2):
				b = s1.copy()
				b.append(s2[-1])
				ret.append(b)
			else:
				break
	return ret


def apriori(filepath, minFrequency):
	"""Runs the apriori algorithm on the specified file with the given minimum frequency"""
	# TODO: implementation of the apriori algorithm

	# Dataset object
	ds = Dataset(filepath)

	# Current sets working
	working_set = [[i] for i in ds.items]

	# Number of frequent sets at the previous level
	previous_frequent = 1

	for level in range(ds.items_num()):  # Check each level
		# Frequent set
		frequent = []

		# Monotonicity property
		if previous_frequent == 0:
			break
		previous_frequent = 0
		# Should not be useful
		if len(working_set) == 0:  # No more frequent set
			break
		for subset in working_set:
			frequency = 0
			for i, set in enumerate(ds.transactions):
				# If the remaining number of transactions is lower than the required frequency
				# PROBLEM COULD COME FROM HERE MAYBE => INDEXES => I DON'T THINK SO
				if ds.trans_num() - i - 1 < minFrequency - frequency:
					break  # Useless to continue
				frequency += is_subset(subset, set)
			if frequency >= minFrequency:
				previous_frequent += 1
				frequent.append(subset)
				print(list(subset), "({})".format(frequency/ds.trans_num()))
		#working_set = gen_supersets_naive(ds, level)
		working_set = gen_supersets_prefix(ds, level, frequent)


def alternative_miner(filepath, minFrequency):
	"""Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
	# TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
	apriori(filepath, minFrequency)


apriori("../Datasets/retail.dat", 1000)
