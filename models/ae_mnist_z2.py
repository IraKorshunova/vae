import cPickle
import gzip
from collections import namedtuple
import lasagne
import theano.tensor as T

batch_size = 100
nhidden = 400
nonlin_enc = T.tanh
nonlin_dec = T.tanh
latent_size = 2
learning_rate_schedule = {
    0: 0.001,
    100: 0.0005,
    200: 0.0001,
    300: 0.00005,
    400: 0.00001
}
max_epoch = 100  # 28
validate_every = 500  # iterations
save_every = 2  # epochs

f = gzip.open('data/mnist.pkl.gz', 'rb')
(x_train, y_train), (x_valid, _), (x_test, _) = cPickle.load(f)
f.close()
ntrain, input_dim = x_train.shape


def build_model():
    # encoder
    l_in = lasagne.layers.InputLayer((batch_size, input_dim))
    l_enc = lasagne.layers.DenseLayer(l_in, num_units=nhidden, nonlinearity=nonlin_enc)

    l_z = lasagne.layers.DenseLayer(l_enc, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity)
    l_mu = l_z
    l_log_sigma = l_z

    # decoder
    l_dec = lasagne.layers.DenseLayer(l_z, num_units=nhidden, nonlinearity=nonlin_dec)
    l_out = lasagne.layers.DenseLayer(l_dec, num_units=input_dim, nonlinearity=lasagne.nonlinearities.sigmoid)

    return namedtuple('Model', ['l_in', 'l_out', 'l_mu', 'l_log_sigma', 'l_dec'])(l_in, l_out, l_mu, l_log_sigma, l_dec)


def build_decoder():
    l_z = lasagne.layers.InputLayer((None, latent_size))
    l_dec = lasagne.layers.DenseLayer(l_z, num_units=nhidden, nonlinearity=nonlin_dec)
    l_out = lasagne.layers.DenseLayer(l_dec, num_units=input_dim, nonlinearity=lasagne.nonlinearities.sigmoid)
    return namedtuple('Decoder', ['l_out', 'l_z'])(l_out, l_z)


def build_objective(model, deterministic=False):
    x_out = lasagne.layers.get_output(model.l_out, deterministic=deterministic)
    x_in = model.l_in.input_var
    log_p_x_given_z = - T.sum(T.nnet.binary_crossentropy(x_out, x_in), axis=1)
    loss = - T.mean(log_p_x_given_z)  # negative lower bound
    return namedtuple('Objective', ['loss'])(loss)
