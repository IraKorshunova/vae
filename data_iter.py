import numpy as np


class DataIterator(object):
    def __init__(self, ntrain, batch_size):
        self.batch_size = batch_size
        self.ntrain = ntrain
        self.rng = np.random.RandomState(42)

    def __iter__(self):
        available_idxs = np.arange(self.ntrain, dtype='int32')
        while len(available_idxs) >= self.batch_size:
            rand_idx = self.rng.choice(range(len(available_idxs)), size=self.batch_size, replace=False)
            yield available_idxs[rand_idx]
            available_idxs = np.delete(available_idxs, rand_idx)