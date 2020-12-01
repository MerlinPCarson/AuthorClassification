import numpy as np
from math import log2
from tqdm import tqdm


def calc_entropy(features, split_idx, epsilon=1e-12):

    num_pos = len(np.where(features[:,0]==1)[0])
    num_neg = len(features) - num_pos

    pr_pos = num_pos/(num_pos+num_neg)
    pr_neg = 1.0 - pr_pos

    # check for domain errors
    if pr_pos > 0:
        e_pos = pr_pos * log2(pr_pos)
    else:
        e_pos = 0

    if pr_neg > 0:
        e_neg = pr_neg * log2(pr_neg)
    else:
        e_neg = 0

    entropy = -(e_pos + e_neg)
    return entropy

def calc_gain(features, split_idx):

    fpos_idxs = np.where(features[:,split_idx] == 1)[0]
    data_fpos = features[fpos_idxs]
    entropy_pos = calc_entropy(data_fpos, split_idx)

    fneg_idxs = np.where(features[:,split_idx] == 0)[0]
    data_fneg = features[fneg_idxs] 
    entropy_neg = calc_entropy(data_fneg, split_idx)

    num_pos = len(data_fpos)

    pr_fpos = num_pos/len(features)
    pr_fneg = 1.0 - pr_fpos

    gain = -(pr_fpos * entropy_pos + pr_fneg * entropy_neg) 
    return gain

def prune_dataset(dataset, n_feats, f_dict):

    # skip first element since features start at 1, and -1.1 > all gains
    gains = [-1.1]
    features = np.array([feats[1] for feats in dataset])
    #print('Calculating information gain for split at each feature')
    for key, value in tqdm(f_dict.items()):
        #if value >= 100:
        #    break
        gains.append(calc_gain(features, value))

    # find features with largest information gain and prune out the rest
    top_feats = sorted(np.argpartition(gains,-n_feats)[-n_feats:])

    #print(f'Gains: {np.array(gains)[top_feats]}')

    # add class idx into features to keep
    top_feats = np.insert(top_feats, 0, 0)
    pruned_features = features[:,top_feats] 
    return pruned_features
