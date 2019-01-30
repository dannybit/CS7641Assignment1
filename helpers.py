from sklearn.model_selection import validation_curve, learning_curve
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from time import clock
from sklearn.model_selection import train_test_split

cv = 5

def make_complexity_curve(clf, x, y,param_name,param_range,clf_name,dataset_name):
    out = defaultdict(dict)
    title = 'Model Complexity Curve: {} - {} ({})'.format(clf_name, dataset_name, param_name)
    train_scores, validation_scores = validation_curve(clf,x,y,param_name,param_range,cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    for pm in param_range:
        out['train'] = train_scores_mean
        out['test'] = validation_scores_mean
    out[param_name] = param_range
    out_df = pd.DataFrame(out)
    out_df.set_index(param_name, inplace=True)
    out_df.name = title
    out_df.to_csv('./output/complexity/{}_{}.csv'.format(title, clock()))
    return out_df

def make_timing_curve(clf, X, Y, clf_name, dataset_name):
    out = defaultdict(dict)
    title = 'Model Timing Curve: {} - {}'.format(clf_name, dataset_name)
    for frac in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=frac, random_state=123)
        train_size = len(X_train)
        st = clock()
        clf.fit(X_train, y_train)
        out['train'][train_size] = clock() - st
        st = clock()
        clf.predict(X_test)
        out['test'][train_size] = clock() - st
    out_df = pd.DataFrame(out)
    out_df.name = title
    out_df.index.name = 'train_size'
    out_df.to_csv('./output/timing/{}_{}.csv'.format(title, clock()))
    return out_df

def make_timing_curve_fixed(clf, X, Y, clf_name, dataset_name):
    out = defaultdict(dict)
    title = 'Model Timing Curve: {} - {}'.format(clf_name, dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=456)
    for frac in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        sample_ids = np.random.choice(X_train.shape[0], int(X_train.shape[0] * frac), replace=False)
        sampled_X_train = X_train[sample_ids, :]
        sampled_y_train = y_train[sample_ids]
        train_size = len(sampled_X_train)
        st = clock()
        clf.fit(sampled_X_train, sampled_y_train)
        out['train'][train_size] = clock() - st
        st = clock()
        clf.predict(X_test)
        out['test'][train_size] = clock() - st
    out_df = pd.DataFrame(out)
    out_df.name = title
    out_df.index.name = 'train_size'
    out_df.to_csv('./output/timing/{}_{}.csv'.format(title, clock()))
    return out_df


def make_learning_curve(clf, x, y,train_sizes,clf_name, dataset_name):
    out = defaultdict(dict)
    title = 'Learning Curve: {} - {}'.format(clf_name, dataset_name)
    train_sizes, train_scores, validation_scores = learning_curve(clf, x, y, cv=cv, train_sizes=train_sizes, shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    out['train_sizes'] = train_sizes
    out['train_scores_mean'] = train_scores_mean
    out['validation_scores_mean'] = validation_scores_mean
    out_df = pd.DataFrame(out)
    out_df.name = title
    out_df.set_index('train_sizes', inplace=True)
    out_df.index.name = 'train_size'
    return out_df

def get_tree_max_depth(x, y):
    clf = DecisionTreeClassifier(max_depth=None)
    clf.fit(x, y)
    return clf.tree_.max_depth