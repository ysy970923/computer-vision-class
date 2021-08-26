import pickle
import numpy as np

def load_data(block_idx, pos_files, neg_files):
    with open(f"../cascade_data/Train/pos/{block_idx}.pkl", 'rb') as f:
        data = pickle.load(f)
    pos_data = [data[f] for f in pos_files]
    with open(f"../cascade_data/Train/neg1/{block_idx}.pkl", 'rb') as f:
        data = pickle.load(f)
    neg_data = [data[f] for f in neg_files]

    train_data = pos_data + neg_data
    labels = np.concatenate((np.full((len(pos_data)), 1), np.full((len(neg_data)), 0)))

    return train_data, labels
    