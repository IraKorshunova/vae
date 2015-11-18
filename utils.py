import cPickle
import theano
import numpy as np

floatX = theano.config.floatX
cast_floatX = np.float32 if floatX == "float32" else np.float64


def save_pkl(obj, path, protocol=cPickle.HIGHEST_PROTOCOL):
    with file(path, 'wb') as f:
        cPickle.dump(obj, f, protocol=protocol)


def load_pkl(path):
    with file(path, 'rb') as f:
        obj = cPickle.load(f)
    return obj
