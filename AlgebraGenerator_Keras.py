from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape, Flatten
import AlgebraGenerator as AG
import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model

class AlgebraLSTMGeneratorKeras():

    def __init__(self, data_dim, timesteps, nb_classes, batch_size, hidden_units, stateful=True, return_sequences=True):

        self.data_dim = 1
        self.timesteps = 1
        self.nb_classes = nb_classes
        self.batch_size = 1

        # expected input batch shape: (batch_size, timesteps, data_dim)
        # note that we have to provide the full batch_input_shape since the network is stateful.
        # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
        print "Build model ----- "
        self.model = Sequential()

        units = hidden_units[0]

        lstm_layer = LSTM(units,
                       return_sequences=return_sequences,
                       stateful=stateful,
                       batch_input_shape=(self.batch_size, self.timesteps, self.data_dim))

        self.model.add(lstm_layer)

        for i in range(1, len(hidden_units)):
            units = hidden_units[i]
            return_sequences = return_sequences if i == len(hidden_units) - 1 else False
            lstm_layer = LSTM(units,
                           return_sequences=return_sequences,
                           stateful=stateful)
            self.model.add(lstm_layer)

        if return_sequences:
            self.model.add(Reshape(target_shape=(hidden_units[-1]*self.timesteps,)))

            self.model.add(Dense(nb_classes, activation='softmax'))

        print "Model shapes:"
        for layer in self.model.layers:
            print layer.output_shape

        self.model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    def train_net(self, epochs, data):
        plt.ion()
        i = 0
        # Create training and testing data
        flattened_data = [item for sublist in data for item in sublist]
        train_data = flattened_data[:-1]
        #labels_data = [item for sublist in data for item in sublist][1:]
        labels_data = flattened_data[1:]
        for epoch in range(0, epochs):
            mean_tr_loss = []
            for i in range(len(train_data)):
                y_true = labels_data[i]
                tr_loss, _ = self.model.train_on_batch(
                    np.expand_dims(np.expand_dims(train_data[i], axis=1), axis=1),
                    np_utils.to_categorical(y_true, 6)
                )
                mean_tr_loss.append(tr_loss)
                i += 1
                plt.scatter(i, tr_loss)
                # plt.show(block=False)
                plt.pause(0.05)
                # plt.show()
            self.model.reset_states()
        plt.waitforbuttonpress()

    def predict(self, seed, N=100):
        for i in range (0, N):
            p = self.model.predict(seed, batch_size=1, verbose=1)
            print p

    def save(self):
        self.model.save('keras_model.h5')

    def load(self):
        self.model = load_model('keras_model.h5')

if __name__ == "__main__":

    ag = AG.AlgebraGenerator(ps=[0.55, 0.25, 0.2], num_trials=3000, fname="AlgebraGenerator.txt")
    if not ag.load():
        ag.run()
        ag.save()
    seq_length = 30
    num_sequences = 20

    # ======== Keras ======== #

    keras_lstm = AlgebraLSTMGeneratorKeras(
        1,
        seq_length,
        nb_classes=6,
        batch_size=100,
        hidden_units=[100, 100],
        return_sequences=True
    )
    keras_lstm.train_net(epochs=300, data=ag.encoded_data)

    test_ag = ag.AlgebraGenerator(ps=[0.55,0.25,0.2], num_trials=1)
    test_ag.run()
    seed = test_ag.encoded_data
    keras_lstm.predict(seed, N=100)