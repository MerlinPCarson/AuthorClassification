#!/usr/bin/python3
# Machine Learner training script for features
# created by features.py
# supported learners include Naive Bayes, XGBoost and Random Forest
# by Merlin Carson

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier as RandomForest
from xgboost import XGBClassifier as XGB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def parse_args():
    parser = argparse.ArgumentParser(description='Feature creator for author classification')
    parser.add_argument('--dataset', type=str, default='features.csv', help='dataset in .csv format')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--algorithm', choices=['RF', 'XGB', 'NB'], default='NB', help='Which algorithm to use (Random Forest, XGBoost, Naive Bayes)')
    parser.add_argument('--num_folds', type=int, default=10, help='Number of folds to train and validate model with')

    return parser.parse_args()

def kfold_validation(features, labels, algorithm, num_folds):

    # split data into n folds for cross validation
    kf = KFold(n_splits=num_folds)
    kf.get_n_splits(features)

    fold_scores = {'train':[], 'val':[]} 

    # perform k-fold cross validation on splits
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(features), start=1):
        print(f'Training on fold {fold_num}')
        X_train, y_train = features[train_idx], labels[train_idx]
        X_val, y_val = features[val_idx], labels[val_idx]

        # create and train Naive Bayes model
        if args.algorithm == 'NB':
            model = BernoulliNB()
            model.fit(X_train, y_train)

        # create and train Random Forest model
        if args.algorithm == 'RF':
            model = RandomForest(n_estimators=100, max_depth=10, n_jobs=os.cpu_count(), verbose=2)
            model.fit(X_train, y_train)

        # create and train XGBoost model
        if args.algorithm == 'XGB':
            model = XGB(verbosity=1, n_estimators=1000, max_depth=3, reg_lambda=1, reg_alpha=1e-4)
            model.fit(X_train, y_train, eval_set=[(X_val,y_val)], eval_metric='logloss', verbose=True, early_stopping_rounds=20)

        # get accuracy on training set
        train_score = model.score(X_train, y_train)
        fold_scores['train'].append(train_score)

        # get accuracy on validation set
        val_score = model.score(X_val, y_val)
        fold_scores['val'].append(val_score)

        print(f'Fold {fold_num}: training score = {train_score}, validation score = {val_score}')

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

    data = pd.read_csv(args.dataset, header=None).to_numpy()
    X = data[:, 2:].astype('uint8')
    y = data[:,1].astype('uint8')

    # perform k-fold cross validation
    if args.num_folds > 1:
        print(f'Performing {args.num_folds}-fold validation')
        f_scores = kfold_validation(X, y, args.algorithm, args.num_folds)
        accs = kfold_scores(f_scores)
        print(f'Average accuracy of {args.num_folds}-folds: {100*accs[0]:.2f}%')
        print(f'Best accuracy of {args.num_folds}-folds: {100*accs[1]:.2f}%')
    else:
        # not performing k-fold, split 15% of training data into validation set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=args.seed)
        print(f'Train data: {X_train.shape}, train labels: {y_train.shape}')
        print(f'Test data: {X_test.shape}, test labels: {y_train.shape}')

        # create and train Naive Bayes model
        if args.algorithm == 'NB':
            model = BernoulliNB()
            model.fit(X_train, y_train)

        # create and train Random Forest model
        if args.algorithm == 'RF':
            model = RandomForest(n_estimators=100, max_depth=10, n_jobs=os.cpu_count(), verbose=2)
            model.fit(X_train, y_train)

        # create and train XGBoost model
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
