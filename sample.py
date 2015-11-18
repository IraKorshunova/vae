import os
import importlib
import string
import time
import cPickle as pickle
from scipy.stats import norm
import theano
import lasagne as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# if not (2 <= len(sys.argv) <= 5):
#     sys.exit("Usage: sample.py <metadata_path>")
#
# metadata_path = sys.argv[1]

metadata_path = 'metadata/vae_mnist_z2-user-20151106-160606.pkl'

with open(metadata_path) as f:
    metadata = pickle.load(f)

config = importlib.import_module('%s' % metadata['configuration'])

target_path = "samples/%s-%s/" % (
    metadata['experiment_id'], time.strftime("%Y%m%d-%H%M%S", time.localtime()))

if not os.path.isdir(target_path):
    os.mkdir(target_path)

decoder = config.build_decoder()
decoder_layers = nn.layers.get_all_layers(decoder.l_out)
print "  decoder layer output shapes:"
nparams = len(nn.layers.get_all_params(decoder.l_out))
nn.layers.set_all_param_values(decoder.l_out, metadata['param_values'][-nparams:])

for layer in decoder_layers:
    name = layer.__class__.__name__
    print "    %s %s" % (string.ljust(name, 32), nn.layers.get_output_shape(layer))

mesh = np.linspace(0.001, 0.999, 20)
z = np.zeros((400, 2), dtype='float32')
for i in xrange(20):
    for j in xrange(20):
        z[20 * i + j, :] = np.array([norm.ppf(mesh[i]), norm.ppf(mesh[j])])

sample = theano.function([decoder.l_z.input_var], nn.layers.get_output(decoder_layers[-1]))

digits = sample(z)

tile = np.zeros((20 * 28, 20 * 28), dtype='float32')

for i in xrange(20):
    for j in xrange(20):
        d = np.reshape(digits[20 * i + j, :], (28, 28))
        tile[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = d

plt.imsave(target_path + 'tile.png', tile, cmap=matplotlib.cm.Greys)
