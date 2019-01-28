import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, source_names, target_names, dimx=200, dimy=200, batch_size=100, n_classes=56, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.dimx = dimx
        self.dimy = dimy
        self.target_names = target_names
        self.source_names = source_names
        self.n_classes = n_classes
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.target_names)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X = np.empty((self.batch_size, *self.dimx))
        y = np.empty((self.batch_size, *self.dimy))

        # Store sample
        X = np.load('../data/sources/' + self.source_names[index] + '.npy')
        # Store output
        y = np.load('../data/targets/' + self.target_names[index] + '.npy')

        return keras.utils.to_categorical(X, num_classes=self.n_classes), \
               keras.utils.to_categorical(y, num_classes=self.n_classes)
