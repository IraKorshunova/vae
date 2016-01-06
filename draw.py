import os
import string
import cPickle as pickle
from scipy.stats import norm
import theano
import lasagne as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import importlib
from matplotlib.patches import Ellipse
from pylab import figure


def encode(metadata, config, target_path):
    model = config.build_model()
    model_layers = nn.layers.get_all_layers(model.l_out)
    nn.layers.set_all_param_values(model.l_out, metadata['param_values'])
    print '  Model'
    for layer in model_layers:
        name = layer.__class__.__name__
        print "    %s %s" % (string.ljust(name, 32), nn.layers.get_output_shape(layer))

    encode = theano.function([model.l_in.input_var],
                             [nn.layers.get_output(model.l_mu), nn.layers.get_output(model.l_log_sigma)])

    x = config.x_train
    enc_out = encode(x)
    mu = enc_out[0]
    log_sigma = enc_out[1]

    np.save(target_path + 'mu.npy', mu)
    np.save(target_path + 'log_sigma.npy', log_sigma)


def plot_codes(metadata, config, target_path):
    if not os.path.isfile(target_path + 'mu.npy') or not os.path.isfile(target_path + 'log_sigma.npy'):
        encode(metadata, config, target_path)

    mu = np.load(target_path + 'mu.npy')
    log_sigma = np.load(target_path + 'log_sigma.npy')

    colors_dict = {0: 'blue', 1: 'green', 2: 'red', 3: 'cyan', 4: 'hotpink',
                   5: 'yellow', 6: 'navy', 7: 'purple', 8: 'lime', 9: 'orange'}
    fig = figure()
    ax = fig.add_subplot(111, aspect='equal')
    if 'vae' in config.__name__:
        sigma = np.exp(log_sigma)
        # ells = [Ellipse(xy=mu[i, :], width=3*sigma[i, 0], height=3*sigma[i, 1], angle=0)
        #         for i in range(len(mu))]
        ells = [Ellipse(xy=mu[i, :], width=0.07, height=0.07, angle=0)
                for i in range(len(mu))]
    else:
        ells = [Ellipse(xy=mu[i, :], width=1, height=1, angle=0)
                for i in range(len(mu))]

    for i, e in enumerate(ells):
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor(colors_dict[config.y_train[i]])

    ax.set_xlim(np.min(mu[:, 0], axis=0), np.max(mu[:, 0], axis=0))
    ax.set_ylim(np.min(mu[:, 1], axis=0), np.max(mu[:, 1], axis=0))
    fig.savefig(target_path + 'codes.png')


def draw_homotopy(metadata, config, target_path, idx1, idx2):
    if not os.path.isfile(target_path + 'mu.npy'):
        encode(metadata, config, target_path)

    mu = np.load(target_path + 'mu.npy')
    x1, y1 = config.x_train[idx1, :], config.y_train[idx1]
    x2, y2 = config.x_train[idx2, :], config.y_train[idx2]
    mu1, mu2 = mu[idx1, :], mu[idx2, :]

    decoder = config.build_decoder()
    decoder_layers = nn.layers.get_all_layers(decoder.l_out)
    nparams = len(nn.layers.get_all_params(decoder.l_out))
    nn.layers.set_all_param_values(decoder.l_out, metadata['param_values'][-nparams:])
    print '  Decoder'
    for layer in decoder_layers:
        name = layer.__class__.__name__
        print "    %s %s" % (string.ljust(name, 32), nn.layers.get_output_shape(layer))

    decode = theano.function([decoder.l_z.input_var], nn.layers.get_output(decoder.l_out))

    p_range = np.arange(1, 0, -0.05)
    tile = np.reshape(x1, (28, 28))

    for p in p_range:
        zp = p * mu1 + (1 - p) * mu2
        zp = zp[np.newaxis, :]
        xp_hat = decode(zp)

        xp_hat = np.reshape(xp_hat, (28, 28))
        tile = np.hstack((tile, xp_hat))

    tile = np.hstack((tile, np.reshape(x2, (28, 28))))

    plt.imsave(target_path + 'homotopy_%s-%s.png' % (str(y1), str(y2)), tile, cmap=matplotlib.cm.Greys)


def draw_tile(metadata, config, target_path):
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


if __name__ == '__main__':
    metadata_path = 'metadata/ae_mnist_z2_l2reg_e150-geit-20160106-103248.pkl'
    # metadata_path = 'metadata/ae_mnist_z2_e100-geit-20160105-120222.pkl'
    # metadata_path = 'metadata/vae_mnist_z2_e150-geit-20160105-172854.pkl'

    with open(metadata_path) as f:
        metadata = pickle.load(f)

    config = importlib.import_module('models.%s' % metadata['configuration'])

    target_path = "samples/%s/" % metadata['experiment_id']
    if not os.path.isdir(target_path):
        os.mkdir(target_path)

    plot_codes(metadata, config, target_path)

    if 'vae' in config.__name__:
        draw_tile(metadata, config, target_path)

    draw_homotopy(metadata, config, target_path, 43045, 21641)  # ae 0-7
    draw_homotopy(metadata, config, target_path, 5895, 19091)  # vae 2-1


    # for i in xrange(10):
    #       for j in xrange(10):
    #           if i != j:
    #               draw_homotopy(metadata, config, target_path, np.where(config.y_train == i)[0][0],
    #                             np.where(config.y_train == j)[0][0])
