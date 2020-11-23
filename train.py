import os
import sys
import time
import pickle
import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier as RandomForest
from xgboost import XGBClassifier as XGB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def parse_args():
    parser = argparse.ArgumentParser(description='Feature creator for author classification')
    parser.add_argument('--text_dir', type=str, default='txts', help='Directory with texts to process for dataset')
    parser.add_argument('--full_ds', type=str, default='full_dataset.npy', help='File to save full feature set to')
    parser.add_argument('--pruned_ds', type=str, default='dataset.npy', help='File to save pruned dataset to')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--all_feats', action='store_true', default=False, help='Use all features')
    parser.add_argument('--algorithm', choices=['RF', 'XGB', 'NB'], default='NB', help='Which algorithm to use (Random Forest, XGBoost, Naive Bayes)')
    parser.add_argument('--num_folds', type=int, default=2, help='Number of folds to train and validate model with')

    return parser.parse_args()

def kfold_validation(features, labels, algorithm='XGB', num_folds=2):

    kf = KFold(n_splits=num_folds)
    kf.get_n_splits(features)

    fold_scores = {'train':[], 'val':[]} 

    fold_num = 0
    for train_idx, val_idx in kf.split(features):
        fold_num += 1
        print(f'Training on fold {fold_num}')
        X_train, y_train = features[train_idx], labels[train_idx]
        X_val, y_val = features[val_idx], labels[val_idx]

        if args.algorithm == 'NB':
            model = BernoulliNB()
            model.fit(X_train, y_train)

        if args.algorithm == 'RF':
            model = RandomForest(n_estimators=100, max_depth=10, n_jobs=os.cpu_count(), verbose=2)
            model.fit(X_train, y_train)

        if args.algorithm == 'XGB':
            model = XGB(verbosity=1, n_estimators=1000, max_depth=8, reg_lambda=1e-2, reg_alpha=4)
            model.fit(X_train, y_train, eval_set=[(X_val,y_val)], eval_metric='logloss', verbose=True, early_stopping_rounds=20)

        train_score = model.score(X_train, y_train)
        fold_scores['train'].append(train_score)

        val_score = model.score(X_val, y_val)
        fold_scores['val'].append(val_score)

        print(f'Fold {fold_num}: training score = {train_score}, validation score = {val_score}')

        with open('fold_accs_random_forest.npy', 'wb') as outfile:
            pickle.dump(fold_scores, outfile)

    return fold_scores

def kfold_scores(f_accs):

    num_folds = len(f_accs['val'])
        
    return sum(f_accs['val'])/num_folds, max(f_accs['val'])

def test_model(model, X, y):
    preds = model.predict(X)
    num_correct = np.sum(preds==y)
    print(f'Test accuracy: {100*num_correct/len(y):.2f}%')

def main(args):
    start = time.time()

    if not args.all_feats:
        data = pickle.load(open(args.pruned_ds, 'rb'))
    else:
        data = pickle.load(open(args.full_ds, 'rb'))
        data = np.array([feats[1] for feats in data])

    X = data[:, 1:]
    y = data[:,0]

    if args.num_folds > 0:
        print(f'Performing {args.num_folds}-fold validation')
        f_scores = kfold_validation(X, y, algorithm=args.algorithm, num_folds=args.num_folds)
        accs = kfold_scores(f_scores)
        print(f_scores)
        print(f'Average accuracy of {args.num_folds}-folds: {100*accs[0]:.2f}%')
        print(f'Best accuracy of {args.num_folds}-folds: {100*accs[1]:.2f}%')
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=args.seed)
        print(f'Train data: {X_train.shape}, train labels: {y_train.shape}')
        print(f'Test data: {X_test.shape}, test labels: {y_train.shape}')

        if args.algorithm == 'NB':
            model = BernoulliNB()
            model.fit(X_train, y_train)

        if args.algorithm == 'RF':
            model = RandomForest(n_estimators=100, max_depth=10, n_jobs=os.cpu_count(), verbose=2)
            model.fit(X_train, y_train)

        if args.algorithm == 'XGB':
            model = XGB(verbosity=1, n_estimators=1000, max_depth=8, reg_lambda=1e-2, reg_alpha=4)
            model.fit(X_train, y_train, eval_set=[(X_test,y_test)], eval_metric='logloss', verbose=True, early_stopping_rounds=20)

        # test model
        test_model(model, X_test, y_test)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
