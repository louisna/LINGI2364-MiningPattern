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

def gen_supersets_naive(ds, level):
	"""
	Naive generation of the supersets, generating all permutations of size level+2
	:param ds: The object representing the dataset
	:param level: The level (~ size) of the sets
	:return: A list of the supersets at the given level
	"""
	return list(itertools.permutations(ds.items, level + 2))

def is_prefix(s1, s2):
	return s1[:-1] == s2[:-1]

def gen_supersets_prefix(sets):
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
	# TODO: implementation of the apriori algorithm

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
			if level == 0 :
				support = len(dico[tuple(subset)]) 
			else:
				trans = dico[tuple(subset[:-1])]
				for i,seti in enumerate(trans):
					# If the remaining number of transactions is lower than the required frequency
					# PROBLEM COULD COME FROM HERE MAYBE => INDEXES => I DON'T THINK SO
					if len(trans) - i < minFrequency * ds.trans_num() - support:
						#print("utilise avec economie", len(trans) - i)
						break  # Useless to continue
					if is_subset(subset, seti):
						support += 1
						freq_trans.append(seti)
			frequency = support / ds.trans_num()
			if frequency >= minFrequency:
				frequent.append(subset)
				if level > 0:
					future_dico[tuple(subset)] = freq_trans
				else:
					future_dico[tuple(subset)] = dico[tuple(subset)]
				print(list(subset), "({})".format(frequency))
		#working_set = gen_supersets_naive(ds, level)
		dico = future_dico
		working_set = gen_supersets_prefix(frequent)


def alternative_miner(filepath, minFrequency):
	"""Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
	# TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
	apriori(filepath, minFrequency)

"""
t=time.time()
apriori("./Datasets/accidents.dat", 0.8)
print(time.time()-t)
"""