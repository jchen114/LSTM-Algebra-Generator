import sys
import numpy as np

maj = sys.version_info

version = 2

if maj[0] >= 3:
	import _pickle as pickle
	import importlib.machinery
	import types
	version = 3
else:
	import cPickle as pickle
	import imp


class AlgebraGenerator():

	# Rules:
	# E -> I
	# E -> M '*' M
	# E -> E '+' E
	# M -> I
	# M -> M '*' M
	# M -> '(' E '+' E ')'

	def __init__(self, ps, num_trials):
		self.ps = ps
		self.num_trials = num_trials
		self.data = list()
		self.encoded_data = list()
		self.char_to_ix = {
			'I': 0,
			'+': 1,
			'*': 2,
			'(': 3,
			')': 4,
			'\n': 5
		}
		self.ix_to_char = {
			0: 'I',
			1: '+',
			2: '*',
			3: '(',
			4: ')',
			5: '\n'
		}

	def run(self):
		for trial in range (0, self.num_trials):
			expression, encoded_data = self.generate()
			self.data.append(expression)
			self.encoded_data.append(encoded_data)
			#print expression

	def generate(self):
		p = np.random.uniform(0.0, 1.0)
		expression = [['E'], ['+'], ['E']] if p < 0.5 else [['M'], ['*'], ['M']]

		while ['E'] in expression or ['M'] in expression:
			try:
				index = expression.index(['E'])
				# sample from the rules
				instance = np.random.choice([
					[['I']],
					[['M'], ['*'], ['M']],
					[['E'], ['+'], ['E']]
				], p=self.ps
				)
				del expression[index]
				expression[index:index] = instance
			except ValueError as err:
				# print "No E left"
				pass
			try:
				index = expression.index(['M'])
				# sample from the rules
				instance = np.random.choice([
					[['I']],
					[['M'], ['*'], ['M']],
					[['('], ['E'], ['+'], ['E'], [')']]
				], p=self.ps
				)
				del expression[index]
				expression[index:index] = instance
			except ValueError as err:
				# print "No M left"
				pass
		expression.append(['\n'])
		encoded_data = [[self.char_to_ix[x[0]]] for x in expression]
		return expression, encoded_data

	def save(self):
		pickle.dump(self.data, open("algebra.p", 'wb'))
		pickle.dump(self.encoded_data, open("algebra-encoded.p","wb"))

	def load(self):
		try:
			self.data = pickle.load(open("algebra.p", "rb"))
			self.encoded_data = pickle.load(open("algebra-encoded.p", "rb"))
			return True
		except Exception as e:
			print(e)
			return False

	def decipher(self, sequence):
		algebra_seq = list()
		sequence = [item for sublist in sequence for item in sublist]
		for el in sequence:
			c = self.ix_to_char[el]
			algebra_seq.append(c)
		print(algebra_seq)


