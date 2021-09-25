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

def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", dest="data", type=str, default=None, help='path to input data')
    parser.add_argument("-o", "--output", dest="output", type=str, default=None, help="path to output folder")
    parser.add_argument("-p", "--cores", "cores", type=int, default=1,
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

    # convert data into DMatrix:
    dt_dmatrix = xgb.DMatrix(data=X, label=y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)
    print("", Counter(y_train), Counter(y_test))
    


