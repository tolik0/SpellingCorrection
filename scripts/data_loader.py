import numpy as np
import keras
import os


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, source_names, target_names, n_classes=56, n_s=64):
        'Initialization'
        self.target_names = target_names
        self.source_names = source_names
        self.n_classes = n_classes
        self.n_s = n_s

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.target_names)

    def __getitem__(self, index):
        'Generate one batch of data'

        # Store sample
        X = np.load('../data/sources/' + self.source_names[index] + '.npy')
        # Store output
        y = np.load('../data/targets/' + self.target_names[index] + '.npy') 
        y = keras.utils.to_categorical(y, num_classes=self.n_classes) # y.shape = (batch_size, T_y, vocab_size)
        y_true = np.concatenate((np.zeros((y.shape[0], 1, y.shape[2])), y), axis=1)[:, :-1, :]

        return ({'X': keras.utils.to_categorical(X, num_classes=self.n_classes),
                    's0': np.zeros((X.shape[0], self.n_s)),
                    'c0': np.zeros((X.shape[0], self.n_s)),
                    'Y_true': y_true},
                 list(y.swapaxes(0, 1)))


def generate_data(path, n_classes, n_s):
    sources = os.listdir(os.path.join(path, "sources\\"))
    targets = os.listdir(os.path.join(path, "targets\\"))
    while True:
        for i in range(len(sources)):
            X = np.load(os.path.join(path, "sources", sources[i]))
            Y = np.load(os.path.join(path, "targets", targets[i]))
            yield ({'X': keras.utils.to_categorical(X, num_classes=n_classes),
                    's0': np.zeros((X.shape[0], n_s)),
                    'c0': np.zeros((X.shape[0], n_s))},
                   {'output': list(keras.utils.to_categorical(Y, num_classes=n_classes).swapaxes(0, 1))})
