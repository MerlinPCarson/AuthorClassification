# ID3 entropy, gain and pruning helper functions
# by Merlin Carson

import numpy as np
from math import log2
from tqdm import tqdm

# calculate entropy of split on feature index
def calc_entropy(features, split_idx):

    # number of each class at feature split
    num_pos = len(np.where(features[:,0]==1)[0])
    num_neg = len(features) - num_pos

    # probability of each class at feature split
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

    # final entropy for feature split
    entropy = -(e_pos + e_neg)
    return entropy

# calculate gain of split on feature index
def calc_gain(features, split_idx):

    # stats for split feature being positive
    fpos_idxs = np.where(features[:,split_idx] == 1)[0]
    data_fpos = features[fpos_idxs]
    entropy_pos = calc_entropy(data_fpos, split_idx)

    # stats for split feature being negative 
    fneg_idxs = np.where(features[:,split_idx] == 0)[0]
    data_fneg = features[fneg_idxs] 
    entropy_neg = calc_entropy(data_fneg, split_idx)

    num_pos = len(data_fpos)

    # probabilities of positive/negative
    pr_fpos = num_pos/len(features)
    pr_fneg = 1.0 - pr_fpos

    # final gain calculation fore feature split 
    gain = -(pr_fpos * entropy_pos + pr_fneg * entropy_neg) 
    return gain

# calculate gain on split at each feature, return pruned feature set
def prune_dataset(dataset, n_feats, f_dict):

    # skip first element since features start at 1, and -1.1 > all gains
    gains = [-1.1]

    # copy features from dataset into numpy array, removes example ID
    features = np.array([feats[1] for feats in dataset])

    #print('Calculating information gain for split at each feature')
    # calculate gain for each feature split
    for key, value in tqdm(f_dict.items()):
        gains.append(calc_gain(features, value))

    # find features with largest information gain and prune out the rest
    top_feats = sorted(np.argpartition(gains,-n_feats)[-n_feats:])

    #print(f'Gains: {np.array(gains)[top_feats]}')

    # add class idx into features to keep
    top_feats = np.insert(top_feats, 0, 0)

    # copy all examples with just the top features
    pruned_features = features[:,top_feats] 
    return pruned_features
