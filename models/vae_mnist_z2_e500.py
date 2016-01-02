import cPickle
import gzip
from collections import namedtuple
import lasagne
import theano.tensor as T
from extra_layers import SamplingLayer

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
max_epoch = 500
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

    l_mu = lasagne.layers.DenseLayer(l_enc, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity)
    l_log_sigma = lasagne.layers.DenseLayer(l_enc, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity)

    # sample z
    l_z = SamplingLayer(mu=l_mu, log_sigma=l_log_sigma)

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
    mu = lasagne.layers.get_output(model.l_mu, deterministic=deterministic)
    log_sigma = lasagne.layers.get_output(model.l_log_sigma, deterministic=deterministic)
    dkl = T.mean(0.5 * T.sum(1 + 2 * log_sigma - mu ** 2 - T.exp(2 * log_sigma), axis=1))
    log_p_x_given_z = - T.mean(T.sum(T.nnet.binary_crossentropy(x_out, x_in), axis=1))
    loss = - dkl - log_p_x_given_z
    return namedtuple('Objective', ['loss', 'kl', 'ce'])(loss, dkl, log_p_x_given_z)

