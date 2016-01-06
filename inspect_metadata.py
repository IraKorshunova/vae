import cPickle as pickle

metadata_path = 'metadata/ae_mnist_z2-koe-20151202-150344.pkl'

with open(metadata_path) as f:
    metadata = pickle.load(f)

print metadata['epochs_since_start']

for i, lv in enumerate(metadata['losses_eval_valid']):
    print i, lv