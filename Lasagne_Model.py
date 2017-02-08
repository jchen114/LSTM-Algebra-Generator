import lasagne
import theano.tensor as T
import numpy as np
import theano
import matplotlib.pyplot as plt

import AlgebraGenerator as AG


class AlgebraLSTMGeneratorLasgne():

	def __init__(self, hidden_units, num_classes, batch_size=1, seq_length=20, num_features=1, learning_rate=.01, ret_final=True):
		lasagne.random.set_rng(np.random.RandomState(1))

		sequences = T.itensor3('sequences')
		target_values = T.ivector('target_output')

		self.seq_length = seq_length
		self.batch_size = batch_size
		self.ret_final = ret_final

		# Constructing network
		# shape = (batches, sequence length, number of features in sequence)
		print("-- Building Network --")

		example = np.random.randint(size=(25, 5, 1), low=0, high=5).astype(dtype='int32')

		l_in = lasagne.layers.InputLayer(shape=(None, None, num_features), input_var=sequences)

		print("after input: ")
		print(lasagne.layers.get_output(l_in).eval({sequences: example}).shape)

		for layer in range(len(hidden_units)):
			return_final = ret_final if (layer == len(hidden_units) - 1) else False
			units = hidden_units[layer]
			l_rnn = lasagne.layers.LSTMLayer(l_in, num_units=units, nonlinearity=lasagne.nonlinearities.tanh,
											 only_return_final=return_final)
			l_do = lasagne.layers.DropoutLayer(l_rnn, p=0.01)
			print("After LSTM + Dropout: ")
			print(lasagne.layers.get_output(l_do).eval({sequences: example}).shape)
			l_in = l_do

		if not ret_final:
			l_in = lasagne.layers.ReshapeLayer(l_in, (-1, hidden_units[-1]))
			print("After reshape: ")
			print(lasagne.layers.get_output(l_in).eval({sequences: example}).shape)

		self.l_out = lasagne.layers.DenseLayer(l_in, num_units=num_classes, W=lasagne.init.Normal(),
											   nonlinearity=lasagne.nonlinearities.softmax)

		print("After output: ")
		print(lasagne.layers.get_output(self.l_out).eval({sequences: example}).shape)

		self.network_output = lasagne.layers.get_output(self.l_out)

		self.cost = lasagne.objectives.categorical_crossentropy(self.network_output, target_values).mean()

		self.all_params = lasagne.layers.get_all_params(self.l_out, trainable=True)

		# updates = lasagne.updates.adagrad(self.cost, self.all_params, learning_rate)
		updates = lasagne.updates.rmsprop(self.cost, self.all_params, learning_rate)
		#updates = lasagne.updates.apply_nesterov_momentum(self.cost, self.all_params, learning_rate)

		self._seq_shared = theano.shared(
			np.zeros((batch_size, seq_length, num_features), dtype='int32')
		)

		self._targets_shared = theano.shared(
			np.zeros((batch_size), dtype='int32')
		)

		self._predict_shared = theano.shared(
			np.zeros((1, seq_length, num_features), dtype='int32')
		)

		givens = {
			sequences: self._seq_shared,
			target_values: self._targets_shared
		}

		self.train = theano.function(
			inputs=[],
			outputs=[self.cost, self.network_output],
			updates=updates,
			givens=givens,
			allow_input_downcast=True
		)

		self.compute_cost = theano.function(
			inputs=[],
			outputs=self.cost,
			givens=givens,
			allow_input_downcast=True
		)

		self.predict = theano.function(
			inputs=[],
			outputs=self.network_output,
			givens={
				sequences: self._predict_shared
			}
		)

	def train_net_multiple_seq(self, num_batches, epochs, gen):
		plt.ion()
		i = 0
		#num_sequences = (0.3 - (-1))/ 0.01 - self.seq_length # Train on entire function
		for epoch in range(0, epochs):
			for batch in range(0, num_batches):
				# build a batch
				data = list()
				labels = list()
				seq = next(gen)
				for (sub_seq, prediction) in divide_sequences(seq, self.seq_length):
					sub_seq = np.array(sub_seq)
					data.append(sub_seq)
					if not self.ret_final:
						for el in sub_seq[1:]:
							labels.extend(el)
					labels.extend(prediction)
				self._seq_shared.set_value(data)
				self._targets_shared.set_value(labels)
				# Train a batch
				cost, ps = self.train()
				# costs.append(cost)
				# print ps
				i += 1
				plt.scatter(i, cost)
				#plt.show(block=False)
				plt.pause(0.05)
				# plt.show()
		plt.waitforbuttonpress()

	def train_net_single_seq(self, num_times, data_gen):
		plt.ion()
		i = 0
		for time in range (0, num_times):
			expression, encoded_data = data_gen()
			inputs = encoded_data[:-1]
			targets = encoded_data[1:]
			targets = [item for sublist in targets for item in sublist]
			self._seq_shared.set_value([inputs])
			self._targets_shared.set_value(targets)
			cost, ps = self.train()
			# costs.append(cost)
			# print ps
			i += 1
			plt.scatter(i, cost)
			# plt.show(block=False)
			plt.pause(0.05)
			# plt.show()
		plt.waitforbuttonpress()

	def predict_net(self, predictions):
		preds = list()
		for x in np.arange(predictions):
			seq = np.random.choice([0, 3], p=[0.5, 0.5])
			p = seq
			seq = [[seq]]
			while p != 5:
				self._predict_shared.set_value([seq])
				ps = self.predict()[-1]
				p = np.random.choice([0,1,2,3,4,5], p=ps) # Select from the distribution
				new_seq = seq
				new_seq.append([p])
				seq = new_seq
			preds.append(seq)
		# Just for kicks...
		seq = [[3], [3], [3]]
		p = 3
		print("Feeding 3 open parenthesis: ")
		while p!= 5:
			self._predict_shared.set_value([seq])
			ps = self.predict()[-1]
			p = np.random.choice([0,1,2,3,4,5], p=ps)
			new_seq = seq
			new_seq.append([p])
			seq = new_seq
		preds.append(seq)
		return preds

	def save(self):
		# Save the parameters into pickles
		np.savez('las_algebra_generator_model.npz', *lasagne.layers.get_all_param_values(self.l_out))

	def load(self):
		try:
			# Load the parameters into pickles
			with np.load('las_algebra_generator_model.npz') as f:
				param_values = [f['arr_%d' % i] for i in range(len(f.files))]
				lasagne.layers.set_all_param_values(self.l_out, param_values)
			return True
		except IOError as e:
			print(e.message)
			return False


def seq_generator(data, length, random=True):
	flattened_data = [item for sublist in data for item in sublist]
	ptr = 0
	while True:
		if random:
			ptr = np.random.randint(0, len(flattened_data) - length)
		else:
			ptr += 1
			if ptr > len(flattened_data) - length:
				ptr = 0
		yield flattened_data[ptr:ptr + length]


def divide_sequences(seq, seq_length):
	while len(seq) > seq_length:
		yield seq[:seq_length], seq[seq_length]
		seq = seq[1:]


if __name__ == "__main__":

	#theano.config.floatX = 'float32'
	ag = AG.AlgebraGenerator(ps=[0.55, 0.25, 0.2], num_trials=3000)
	if not ag.load():
		ag.run()
		ag.save()
	seq_length = 30
	num_sequences = 20
	generator = seq_generator(ag.encoded_data, seq_length+num_sequences)

	# ======== Lasagne ======== #

	net = AlgebraLSTMGeneratorLasgne([100, 100], 6, seq_length=seq_length, ret_final=False)
	if not net.load():
		#net.train_net_multiple_seq(500, 10, generator)
		net.train_net_single_seq(1500, ag.generate)
		net.save()
	_, seed = ag.generate()
	predictions = net.predict_net(10)
	for prediction in predictions:
		ag.decipher(prediction)