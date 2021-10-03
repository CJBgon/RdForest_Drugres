#!/usr/bin/env python

# Perform a bayesian CV
# data is loaded in (first column = index, second column is outcome (binary)
# train test split formed into a xgboost compatible DMatrix
# parameters are optimized with Bayes_opt
# final model is ran and metrics are output.

import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import joblib
from bayes_opt import BayesianOptimization
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", dest="data", type=str, default=None, help='path to input data')
    parser.add_argument("-o", "--output", dest="output", type=str, default=None, help="path to output folder")
    parser.add_argument("-c", "--cores", dest="cores", type=int, default=-1,
                        help="Number of cores to use for parallel processing")
    parser.add_argument("-i", "--it", dest="it", type=int, default=500,
                        help="Number of iterations during bayesian optimisation")
    parser.add_argument("-r", "--rand", dest="rand", type=int, default=50,
                        help="Number of random starting points to initiate bayesian optimisation")

    options = parser.parse_args()
    if not options.data:
        parser.error("no input data was provided")
    if not options.output:
        parser.error("no output path provided")

    return options


def datload(dat):
    dt = pd.read_csv(dat, index_col=0)
    dt.dropna(axis=0, inplace=True)
    y, X = dt.iloc[:, 0], dt.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print("training set value counts: %s, testing set value counts: %s"%(Counter(y_train), Counter(y_test)))
    return X_train,X_test,y_train,y_test


# Bayesian Optimization function for xgboost
# specify the parameters you want to tune as keyword arguments
# some parameters we want to kep as set (like the booster :'tree')


def save_bay_model(dtrain,
                   num_it,
                   num_rand,
                   error_metric='error'
                   ):
    """
    train xgboost model with bayesian optimization of hyperparameters
    :param dtrain: xgboost compatible DMatrix with training data
    :param num_it: int. number of iterations of bayesian optimzation
    :param num_rand: int. number of random starting points in bayesian optimization
    :param error_metric: str. type of error metric to use in xgb.cv
    :return: optimized hyperparameters for xgboost gbtree
    """

    single_param = {'booster': 'gbtree',
                    'objective':'binary:hinge',
                    'eval_metric': error_metric,
                    'verbosity':0,
                    }

    maxboost = 1500
    early_stop = 20

    def xgb_fun(max_depth,
                subsample,
                    gamma,
                    n_estimators,
                    learning_rate,
                    min_child_weight,
                    max_delta_step,
                    reg_alpha,
                    reg_lambda,
                    base_score,
                    colsample_bytree):
        # the xgb function has to be nested in the bayesian optimisation because
        # BayesianOptimisation does not include other parameters aside from the function and
        # pbound parameters. Therefore the training data (Dtrain in this case) has to be
        # in the function namespace.

            param_dict = {'n_estimators':int(n_estimators),
                          'subsample':subsample,
                          'learning_rate': learning_rate,
                          'max_depth': int(max_depth),
                          'min_child_weight': int(min_child_weight),
                          'max_delta_step': int(max_delta_step),
                          'gamma': gamma,
                          'reg_alpha': reg_alpha,
                          'reg_lambda': reg_lambda,
                          'base_score': base_score,
                          'colsample_bytree': colsample_bytree
                          }

            # add the non-variable parameters
            param_dict.update(single_param)


            xgb_mod = xgb.cv(params=param_dict,
                             nfold=3,
                             stratified=True,
                             dtrain=dtrain,
                             metrics='error',
                             num_boost_round=maxboost,
                             early_stopping_rounds=early_stop,
                             verbose_eval=False)

            return xgb_mod.iloc[0,2]

    optimize = BayesianOptimization(f=xgb_fun,
                                    pbounds={'n_estimators': (3, 1500),
                                             'subsample': (0,1),
                                             'learning_rate': (0.01, 0.8),
                                             'max_depth': (2, 100),
                                             'min_child_weight': (1, 20),
                                             'max_delta_step': (1, 10),
                                             'gamma': (0.2, 20),
                                             'reg_alpha': (0.01, 1),
                                             'reg_lambda': (0.01, 10),
                                             'base_score': (0.1, 1  ),
                                             'colsample_bytree': (0.6, 1)}
                                    )
    optimize.maximize(n_iter=num_it, init_points=num_rand)

    print('\nbest fit:', optimize.max)
    return optimize.max



def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    options = Parser()

    X_train, X_test, y_train, y_test = datload(dat= options.data)

    # transform numpy arrays into dmatrix format
    train = xgb.DMatrix(X_train, label=y_train)
    test = xgb.DMatrix(X_test, label=y_test)

    maxbay = save_bay_model(dtrain=train,
                   error_metric='error',
                   num_rand=options.rand,
                   num_it=options.it)


    model = xgb.XGBClassifier(objective='binary:hinge',
                              parameters = maxbay['params'])

    model.fit(X_train,y_train)
    print('###### final model evaluation #####')
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
    outputj = options.output+'model.joblib'
    joblib.dump(model, filename=outputj, compress=9)
    # export  classification reports:
    test_report = pd.DataFrame(classification_report(y_test, y_test_hat, output_dict=True)).transpose()
    test_report.to_csv(options.output+'/classification_metrics.csv')

if  __name__ == "__main__":
    main()



