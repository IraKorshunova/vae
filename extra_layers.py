import lasagne
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class SamplingLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, log_sigma, **kwargs):
        super(SamplingLayer, self).__init__([mu, log_sigma], **kwargs)

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        mu, log_sigma = input
        eps = self._srng.normal(mu.shape)
        z = mu + T.exp(log_sigma) * eps
        return z
