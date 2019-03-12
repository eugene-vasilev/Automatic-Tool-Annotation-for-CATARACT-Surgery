import numpy as np
from keras.utils import Sequence


class Seq(Sequence):

    def __init__(self, x_set, batch_size, process_func):
        self.x = x_set
        self.batch_size = batch_size
        self.process_func = process_func
        self.indices = np.arange(len(self.x))

    def __len__(self):
        return (len(self.x) // self.batch_size)

    def __getitem__(self, idx):
        batch_x, batch_y = self.process_func(self.x[idx * self.batch_size:(idx + 1) * self.batch_size])

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        self.x = self.x[self.indices]
