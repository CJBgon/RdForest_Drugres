#!/usr/bin/env python

# Perform a grid search CV on imbalanced dataset.
# data is loaded in (first column = index, second column is outcome (binary)
# train test split
# class balancing with SMOTE
# grid searchCV with xgboost is performed

import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline

def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", dest="data", type=str, default=None, help='path to input data')
    parser.add_argument("-o", "--output", dest="output", type=str, default=None, help="path to output folder")
    parser.add_argument("-c", "--cores", dest="cores", type=int, default=1,
                        help="Number of cores to use for parallel processing")

    options = parser.parse_args()
    if not options.data:
        parser.error("no input data was provided")
    if not options.output:
        parser.error("no output path provided")

    return options

def datload(dat):
    dt = pd.read_csv(dat, index_col=0)
    dt.dropna(axis=0, inplace=True)  # dropping patient PTP169-1, NA on the outcome.
    y, X = dt.iloc[:, 0], dt.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print("training set value counts: %s, testing set value counts: %s"%(Counter(y_train), Counter(y_test)))
    return X_train,X_test,y_train,y_test

def training(X_train, y_train, param_grid, p):

    stratified_kfold = StratifiedKFold(n_splits=3,
                                       shuffle=True,
                                       random_state=345)

    pipeline_xgb = imbpipeline(steps=[['smote', SMOTE(random_state=11)],
                                      ['xgb', xgb.XGBClassifier()]
                                      ])

    grid_search_xgb = GridSearchCV(estimator=pipeline_xgb,
                                   param_grid=param_grid,
                                   scoring='f1',
                                   cv=stratified_kfold,
                                   n_jobs=p)

    grid_search_xgb.fit(X_train, y_train)
    return grid_search_xgb


# main:
def main():
    options = Parser()

    parameters = {'smote__sampling_strategy': [0.6, 0.7, 0.8],
                  'smote__k_neighbors': [3, 4, 5],
                  'xgb__n_estimators': [5, 10, 100, 200],
                  'xgb__learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
                  'xgb__max_depth': [2, 4, 6, 10],
                  'xgb__min_child_weight': [1, 2, 5],
                  'xgb__max_delta_step': [1,5,7],
                  'xgb__booster': ['gbtree', 'gblinear', 'dart'],
                  'xgb__eval_metric': ['auc'],
                  'xgb__gamma': [0, 0.2, 0.5, 0,8, 1],
                  'xgb__reg_alpha': [0, 0.1, 0.3, 0.5, 0.7, 1],
                  'xgb__reg_lambda': [0.1, 0.3,0.5, 1, 5, 10],
                  'xgb__base_score': [0.2, 0.5, 1],
                  'xgb__colsample_bytree': [0.5, 1],
                  }
    X_train, X_test, y_train, y_test = datload(dat = options.data)

    model = training(X_train,
                     y_train,
                     p=options.cores,
                     param_grid=parameters)

    cv_score = model.best_score_
    test_score = model.score(X_test, y_test)


    print('***************************** model evalation ***************************** ')
    print("Parameters: ", model.best_params_)
    print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')

    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)


    print('train performance')
    print('---------------------------------------------')
    print(classification_report(y_train, y_train_hat))

    print('Test performance')
    print('---------------------------------------------')
    print(classification_report(y_test, y_test_hat))

    print('Roc_auc score')
    print('---------------------------------------------')
    print(roc_auc_score(y_test, y_test_hat))
    print('')

    print('Confusion matrix')
    print('---------------------------------------------')
    print(confusion_matrix(y_test, y_test_hat))

    # exporting model and evaluation:
    test_report = pd.DataFrame(classification_report(y_test, y_test_hat, output_dict=True)).transpose()
    outfile = options.output + '/classification_report.csv'
    test_report.to_csv(outfile)

if  __name__ == "__main__":
    main()
