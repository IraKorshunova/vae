import importlib
import os
import platform
import string
import sys
import time
import lasagne
import numpy as np
import theano
import theano.tensor as T
import utils
from data_iter import DataIterator

theano.config.warn_float64 = 'warn'
print
if len(sys.argv) < 2:
    print "Usage: %s <config name>" % os.path.basename(__file__)
    config_name = 'vae_mnist_z2_e30'
else:
    config_name = sys.argv[1]

print "Model:", config_name
config = importlib.import_module("models.%s" % config_name)

expid = "%s-%s-%s" % (config_name, platform.node(), time.strftime("%Y%m%d-%H%M%S", time.localtime()))
print "expid:", expid
print

metadata_target_path = 'metadata/%s.pkl' % expid
if not os.path.isdir('metadata'):
    os.mkdir('metadata')

# build model
model = config.build_model()
all_layers = lasagne.layers.get_all_layers(model.l_out)
all_params = lasagne.layers.get_all_params(model.l_out)
num_params = lasagne.layers.count_params(model.l_out)

print "  number of parameters: %d" % num_params
print "  layer output shapes:"
for layer in all_layers:
    name = layer.__class__.__name__
    print "    %s %s" % (string.ljust(name, 32), lasagne.layers.get_output_shape(layer))
print

# build loss
if hasattr(config, 'build_objective'):
    obj = config.build_objective(model)
    obj_valid = config.build_objective(model, deterministic=True)
else:
    raise NotImplementedError

# build updates
learning_rate = theano.shared(np.float32(0.0))
if hasattr(config, 'build_updates'):
    updates = config.build_updates(obj.loss, all_params, learning_rate)
else:
    updates = lasagne.updates.adam(obj.loss, all_params, learning_rate)

# load data to GPU
xtrain_shared = theano.shared(utils.cast_floatX(config.x_train))
xvalid_shared = theano.shared(utils.cast_floatX(config.x_valid))

idxs = T.ivector('idx')
givens = {model.l_in.input_var: xtrain_shared[idxs]}

train = theano.function([idxs], obj.loss, givens=givens, updates=updates, allow_input_downcast=True)
eval_valid = theano.function([], obj_valid, givens={model.l_in.input_var: xtrain_shared})
eval_train = theano.function([], obj_valid, givens={model.l_in.input_var: xvalid_shared})

train_data_iter = DataIterator(config.ntrain, config.batch_size)

print 'Train model'
print
train_batches_per_epoch = config.ntrain / config.batch_size
max_niter = config.max_epoch * train_batches_per_epoch
losses_train = []
losses_eval_train, losses_eval_valid = [], []
niter = 1
start_epoch = 0
prev_time = time.clock()

for epoch in xrange(start_epoch, config.max_epoch):

    if epoch in config.learning_rate_schedule:
        lr = np.float32(config.learning_rate_schedule[epoch])
        print "  setting learning rate to %.7f" % lr
        print
        learning_rate.set_value(lr)

    for train_batch_idxs in train_data_iter:
        train_loss = train(train_batch_idxs)

        losses_train.append(train_loss)
        niter += 1

        if niter % config.validate_every == 0:
            current_time = time.clock()
            print '%d/%d (epoch %.3f) time/nvalid_iter=%.2fs' % (
                niter, max_niter, niter / float(train_batches_per_epoch), current_time - prev_time)
            prev_time = current_time

            valid_loss = eval_valid()
            losses_eval_valid.append(valid_loss)
            eval_train_loss = eval_valid()
            losses_eval_train.append(eval_train_loss)
            if 'vae' in config.__name__:
                print 'validation loss: %0.4f KL: %0.4f CE: %0.4f' % (valid_loss[0], -valid_loss[1], valid_loss[2])
                print 'train loss: %0.4f KL: %0.4f CE: %0.4f' % (
                    eval_train_loss[0], -eval_train_loss[1], eval_train_loss[2])
            else:
                print 'validation loss: %0.4f' % valid_loss[0]
                print 'train loss: %0.4f' % eval_train_loss[0]
            print

    if (epoch + 1) % config.save_every == 0:
        d = {
            'configuration': config_name,
            'experiment_id': expid,
            'epochs_since_start': epoch,
            'losses_train': losses_train,
            'losses_eval_valid': losses_eval_valid,
            'losses_eval_train': losses_eval_train,
            'param_values': lasagne.layers.get_all_param_values(model.l_out)
        }
        utils.save_pkl(d, metadata_target_path)
        print "  saved to %s" % metadata_target_path
        print
