import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble._forest import RandomForestClassifier


#####################
# load in the data
#####################
preTX_dct = pd.read_excel('/RandomForest/PreTX_dct.xlsx')
preTX_ddct = pd.read_excel('/RandomForest/PreTX_ddct.xlsx')
preTX_dct.columns
ddct_slice = preTX_ddct.iloc[:, np.r_[0, 9:30]]

newcol = []
for col in ddct_slice.columns[1:]:
    newcol.append('ddct_' + col)
ddct_slice.columns = ['Patient'] + newcol

dt = preTX_dct.merge(ddct_slice, how='left', on='Patient')

dt.columns
dt = dt.drop(dt.iloc[:, np.r_[1:4, 5:9]], axis=1)
dt.set_index('Patient', inplace=True)

#####################
# Exploratory data analysis
#####################
dt.describe(include='all').T
# 56 features is quite a lot for ~30 samples.
# sns.set()
# sns.pairplot(dt, kind='reg')
# times out, not enough processing power here?
# plt.show()
dt.corr()
plt.figure(figsize=(40, 40), dpi=80)
sns.heatmap(dt.corr(), annot=True)
plt.show()
# there is some correlation between ddct and dct values.
# remove those with high correlation:
cor_features = dt.corr()[(dt.corr() >= 0.7) & (dt.corr() < 1)]
cor_features.dropna(how='all', inplace=True)
cor_features.dropna(how='all', axis=1, inplace=True)

plt.figure(figsize=(10, 10), dpi=80)
sns.heatmap(cor_features)
plt.show()

rm_feat = ['E-CTC', 'M-CTC', 'FBLN1', 'FOLH1', 'KLK2', 'KLK3', 'KLK4', 'MCAM', 'PRKD1']
dt.drop(rm_feat, inplace=True, axis=1)

# are there features with outliers or skewed distributions?


def plothist(coll, data):
    plt.figure(figsize=(15, 12))

    for i, c in enumerate(coll):
        plt.subplot(5, 2, i+1)
        sns.distplot(data[c])
        plt.title('Distribution plot for field:' + c)
        plt.xlabel('')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


def plotskew(coll, data):
    plt.figure(figsize=(20, 30))
    for i, c in enumerate(coll):
        plt.subplot(10, 2, i*2+1)
        sns.boxplot(data[c], color='blue')
        plt.title('Distribution ot for parameter:' + c)
        plt.xlabel('')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1)

        plt.subplot(10, 2, i*2+2)
        sns.boxplot(data[c].apply('log1p'), color='red')
        plt.title('Log1p distribution plot for parameter' + c)
        plt.xlabel('')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1)
    plt.show()


plotskew(coll=dt.columns[0:5], data=dt)
plothist(coll=dt.columns[0:5], data=dt)

#################################################
# create train and test set for cross-validation
#################################################
# seperate target and covariates:
dt.dropna(axis=0, inplace=True)  # dropping patient PTP169-1, NA on the outcome.
y, X = dt.iloc[:, 0], dt.iloc[:, 1:]

# convert data into DMatrix:
dt_dmatrix = xgb.DMatrix(data=X, label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)
Counter(y_train)  # 19 no prog, 7 prog
Counter(y_test)  # 5 no prob, 2 prog

#################################################
# set up pipeline for models and SMOTE
#################################################
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
# set a class weight to increase precision.

pipeline_rf = imbpipeline(steps=[['smote', SMOTE(random_state=11)],
                                 ['rf', RandomForestClassifier()]
                                 ])

#################################
# Train and evaluate
#################################
stratified_kfold = StratifiedKFold(n_splits=3,
                                   shuffle=True,
                                   random_state=345)
param_grid = {'smote__sampling_strategy': [0.6, 0.7, 0.8],
              'smote__k_neighbors': [3, 4, 5],
              'rf__n_estimators': [5, 10, 50, 100],
              'rf__max_depth': [3, 5, 10, 20, 25, None],
              'rf__min_samples_leaf': [1, 2, 3, 4, 5],
              'rf__class_weight': ["balanced"]
              }

grid_search_rf = GridSearchCV(estimator=pipeline_rf,
                              param_grid=param_grid,
                              scoring='f1',
                              cv=stratified_kfold,
                              n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
cv_score = grid_search_rf.best_score_
test_score = grid_search_rf.score(X_test, y_test)
print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')


y_train_hat = grid_search_rf.predict(X_train)
y_test_hat = grid_search_rf.predict(X_test)


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


###########################
# plot ROC and tree metrics
###########################
from sklearn.metrics import plot_roc_curve
from sklearn import tree

plt.figure(figsize=(12,12),dpi=300)
plot_roc_curve(estimator=grid_search_rf,
               X=X_test,
               y=y_test)
plt.savefig('/RandomForest/rf_ROC.png')

forest = grid_search_rf.best_estimator_.named_steps["rf"]
fr_feat = forest.feature_importances_
fr_name = dt.columns[1:]

plt.figure(figsize=(12, 10), dpi=300)
feature_imp = pd.Series(fr_feat, index=fr_name).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualising important features for RandomForestClassifier')
plt.savefig('/RandomForest/rf_features.png')

fn = dt.columns[1:]
cn = dt.columns[0]
fig, axes = plt.subplots(nrows=1, ncols=len(forest.estimators_), figsize=(10, 2), dpi=900)
for index in range(0, len(forest.estimators_)):
    tree.plot_tree(forest.estimators_[index],
                   feature_names=fn,
                   class_names=cn,
                   filled = True,
                   ax=axes[index])
    axes[index].set_title('Estimator: ' + str(index), fontsize =11)
plt.savefig('/RandomForest/rf_trees.png')

###############################
# export the model with joblib
###############################
import joblib
joblib.dump(forest, '/RandomForest/forest_model.pkl', compress=9)

# export  classification reports:
test_report = pd.DataFrame(classification_report(y_test, y_test_hat, output_dict=True)).transpose()
test_report.to_csv('/RandomForest/classification_metrics.csv')



#############################
# Set up XGBoost classifier
#############################

param_grid = {'smote__sampling_strategy': [0.6, 0.7, 0.8],
              'smote__k_neighbors': [3, 4, 5],
              'xgb__n_estimators': [10, 100, 200],
              'xgb__learning_rate': [0.01, 0.05, 0.1],
              'xgb__booster': ['gbtree', 'gblinear'],
              'xgb__gamma': [0, 0.5, 1],
              'xgb__reg_alpha': [0, 0.5, 1],
              'xgb__reg_lambda': [0.5, 1, 5],
              'xgb__base_score': [0.2, 0.5, 1]
              }

pipeline_xgb = imbpipeline(steps=[['smote', SMOTE(random_state=11)],
                                  ['xgb', xgb.XGBClassifier(objective="binary:logistic")]
                                  ])

grid_search_xgb = GridSearchCV(estimator=pipeline_xgb,
                              param_grid=param_grid,
                              scoring= 'f1',
                              cv=stratified_kfold,
                              n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)
cv_score = grid_search_xgb.best_score_
test_score = grid_search_xgb.score(X_test, y_test)
print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')

y_train_hat = grid_search_xgb.predict(X_train)
y_test_hat = grid_search_xgb.predict(X_test)


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
# model has the same test performance as the random forest.
