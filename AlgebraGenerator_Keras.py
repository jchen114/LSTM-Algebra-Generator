from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape, Flatten, Masking, Dropout, TimeDistributed
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

import AlgebraGenerator as AG
import matplotlib.pyplot as plt
import numpy as np

import threading

import os

import time

from keras.models import load_model

class AlgebraLSTMGeneratorKeras():

	def __init__(self, max_seq_length, num_features, nb_classes, lstm_layers ):

		files = [f for f in os.listdir('.') if os.path.isfile(f)]
		self.loaded = False
		for f in files:
			if f.startswith('keras_model'):
				print(f)
				self.model = load_model(f)
				self.loaded = True
				break

		if not self.loaded:
			self.data_dim = 1
			self.timesteps = 1
			self.nb_classes = nb_classes
			self.batch_size = 1

			# expected input batch shape: (batch_size, timesteps, data_dim)
			# note that we have to provide the full batch_input_shape since the network is stateful.
			# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
			print("Build model ----- ")
			self.model = Sequential()

			self.model.add(
				Masking(
					mask_value=-1.0,
					input_shape=(max_seq_length, num_features)
				)
			)

			for layer in lstm_layers:
				self.model.add(
					LSTM(
						output_dim=layer,
						return_sequences=True
					)
				)
				self.model.add(
					Dropout(
						p=0.2
					)
				)

			# Output Layer
			self.model.add(
				TimeDistributed(
					Dense(
						activation='softmax',
						output_dim=nb_classes
					)
				)
			)

			start = time.time()

			self.model.compile(
				optimizer='rmsprop',
				loss='categorical_crossentropy'
			)

			print("Compilation Time : ", time.time() - start)

			print('model layers: ')
			print(self.model.summary())

			print('model.inputs: ')
			print(self.model.input_shape)

			print('model.outputs: ')
			print(self.model.output_shape)

	def train_net(self, epochs, inputs, labels, batch_size=32):

		filepath = "keras_model-{epoch:02d}-{val_loss:.5f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
		earlyStopping = EarlyStopping(
			monitor='val_loss',
			patience=20
		)
		callbacks = [checkpoint, earlyStopping]

		self.model.fit(
			x=np.asarray(inputs),
			y=np.asarray(labels),
			batch_size=batch_size,
			nb_epoch=epochs,
			validation_split=0.1,
			callbacks=callbacks
		)

	def train_on_generator(self, gen, epochs, batch_size=32):
		filepath = "keras_model-{epoch:02d}-{val_loss:.5f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
		earlyStopping = EarlyStopping(
			monitor='val_loss',
			patience=20
		)
		callbacks = [checkpoint, earlyStopping]

		self.model.fit_generator(
			generator=gen,
			samples_per_epoch=batch_size * 100,
			nb_epoch=epochs,
			callbacks=callbacks,
			validation_data=gen,
			nb_val_samples=15,
			nb_worker=10
		)

	def predict(self, seq_length, N=20):
		completed_sequences = list()
		for _ in np.arange(N):
			sequence = -1.0 * np.ones(shape=(seq_length, 1))
			seed = np.random.choice([0, 3], p=[0.5, 0.5])
			sequence[0][:] = seed
			p = seed
			index = 0
			while p != 5:
				prediction = self.model.predict_on_batch(
					x=np.asarray([sequence])
				)
				ps = prediction[0][index]
				p = np.random.choice([0, 1, 2, 3, 4, 5], p=ps)  # Select from the distribution
				index += 1
				sequence[index][0] = p
			sequence = [el for el in sequence if el != -1]
			completed_sequences.append(sequence)

		# Just for kicks...

		p = 3
		sequence = -1.0 * np.ones(shape=(seq_length, 1))
		seed = [[3], [3], [3]]
		sequence[:3] = seed
		index = 2
		while p != 5:
			prediction = self.model.predict_on_batch(
				x=np.asarray([sequence])
			)
			ps = prediction[0][index]
			p = np.random.choice([0, 1, 2, 3, 4, 5], p=ps)
			index += 1
			sequence[index][0] = p
		sequence = [el for el in sequence if el != -1]
		completed_sequences.append(sequence)
		return completed_sequences


def build_input_labels(data, max_seq_length, num_features, nb_classes):
	inputs = list()
	labels = list()
	for datum in data:
		input = -1 * np.ones(shape=(max_seq_length, num_features))
		label = 5 * np.ones(shape=(max_seq_length, 1))

		input[:len(datum)-1] = datum[:-1]
		label[:len(datum)-1] = datum[1:]
		label = to_categorical(label, nb_classes=nb_classes)

		inputs.append(input)
		labels.append(label)
	return inputs, labels


def generator(ag, max_seq_length, sample_size=32):
	while True:
		samples = list()
		targets = list()
		while len(samples) < sample_size:
			expression, encoded_data = ag.generate()
			while len(expression) > max_seq_length:
				expression, encoded_data = ag.generate()
			inputs, labels = build_input_labels([encoded_data], max_seq_length, 1, 6)
			samples.extend(inputs)
			targets.extend(labels)
		yield (np.asarray(samples), np.asarray(targets))


class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
		serializing call to the `next` method of given iterator/generator.
		"""

	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return self.it.next()



if __name__ == "__main__":

	ag = AG.AlgebraGenerator(ps=[0.55, 0.25, 0.2], num_trials=5000)
	if not ag.load():
		ag.run()
		ag.save()
	seq_length = 1800

	# ======== Keras ======== #

	keras_lstm = AlgebraLSTMGeneratorKeras(
		max_seq_length=seq_length,
		num_features=1,
		nb_classes=6,
		lstm_layers=[128, 128]
	)

	gen = generator(ag, seq_length)
	gen = threadsafe_iter(gen)

	#keras_lstm.train_on_generator(gen, 30)
	#inputs, labels = build_input_labels(ag.encoded_data, seq_length, 1, 6)
	#keras_lstm.train_net(epochs=30, inputs=inputs, labels=labels, batch_size=32)

	# test_ag = AG.AlgebraGenerator(ps=[0.55,0.25,0.2], num_trials=1)
	# test_ag.run()
	# seed = test_ag.encoded_data
	sequences = keras_lstm.predict(seq_length, N=20)
	for sequence in sequences:
		ag.decipher(sequence)