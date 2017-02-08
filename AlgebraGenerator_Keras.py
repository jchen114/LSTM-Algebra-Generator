from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape, Flatten, Masking, Dropout, TimeDistributed
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

import AlgebraGenerator as AG
import matplotlib.pyplot as plt
import numpy as np

import time

from keras.models import load_model

class AlgebraLSTMGeneratorKeras():

	def __init__(self, max_seq_length, num_features, nb_classes, lstm_layers ):

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

	def predict(self, seed, N=100):
		preds = list()
		for x in np.arange(N):
			seq = np.random.choice([0, 3], p=[0.5, 0.5])
			p = seq
			seq = [[seq]]
			while p != 5:
				prediction = self.model.predict_on_batch(
					x=np.asarray([seq])
				)
				ps = prediction[-1]
				p = np.random.choice([0, 1, 2, 3, 4, 5], p=ps)  # Select from the distribution
				new_seq = seq
				new_seq.append([p])
				seq = new_seq
			preds.append(seq)
		# Just for kicks...
		seq = [[3], [3], [3]]
		p = 3
		print("Feeding 3 open parenthesis: ")
		while p != 5:
			predictions = self.model.predict_on_batch(
				x=np.asarray([seq])
			)
			ps = predictions[-1]
			p = np.random.choice([0, 1, 2, 3, 4, 5], p=ps)
			new_seq = seq
			new_seq.append([p])
			seq = new_seq
		preds.append(seq)
		return preds

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


if __name__ == "__main__":

	ag = AG.AlgebraGenerator(ps=[0.55, 0.25, 0.2], num_trials=5000)
	if not ag.load():
		ag.run()
		ag.save()
	seq_length = 1500

	# ======== Keras ======== #

	keras_lstm = AlgebraLSTMGeneratorKeras(
		max_seq_length=seq_length,
		num_features=1,
		nb_classes=6,
		lstm_layers=[128, 128]
	)

	inputs, labels = build_input_labels(ag.encoded_data, seq_length, 1, 6)

	keras_lstm.train_net(epochs=30, inputs=inputs, labels=labels, batch_size=32)

	# test_ag = AG.AlgebraGenerator(ps=[0.55,0.25,0.2], num_trials=1)
	# test_ag.run()
	# seed = test_ag.encoded_data
	# keras_lstm.predict(seed, N=100)