# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 11:31:06 2023

@author: Davood
"""


# import scipy.io
import os
import random
import numpy as np
import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn import preprocessing, model_selection, impute
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, f1_score, classification_report
from imblearn.metrics import classification_report_imbalanced
from sklearn.pipeline import Pipeline
import time
import datetime
import dill
import matplotlib.pyplot as plt
from prepare_data import prepare_data
from xgboost import XGBClassifier


# %% =============================================================================
# prep data
file_path = '10 and older AI, IGE Focal 2023.sav'
feature_names = ["Age_At_Onset", "Sex", "Febrile_Convulsion", "Family_Hx_Of_Epilepsy",
                 "Major_Head_Injury", "Medical_Comorbidity", "Seizure_Aura_1", "Exam"]
label_names = ["FinalDx"]
label_values = ['Focal', 'IGE']
case_indicators = ["Row",	"Number"]
any_description = {"Tongue_Biting": ["tongue", "biting", "Tongue", "Biting"]}

data, case_indicators, label_names, feature_names, enc = prepare_data(
    file_path=file_path, case_indicators=case_indicators, label_names=label_names, label_values=label_values, feature_names=feature_names, any_description=any_description)
[train_index, test_index] = list(model_selection.StratifiedShuffleSplit(
    n_splits=1, test_size=0.3, random_state=0).split(data[feature_names], data[label_names]))[0]

X_train, y_train, X_test, y_test = data[feature_names].iloc[train_index].to_numpy(), data[label_names].iloc[
    train_index].to_numpy(), data[feature_names].iloc[test_index].to_numpy(), data[label_names].iloc[test_index].to_numpy()

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# %% =============================================================================
# multinomial Naive Bayes
# steps = []
# steps.append(("imputer", SimpleImputer(
#     missing_values=np.nan, strategy="constant", fill_value=2)))
# steps.append(('scale', preprocessing.MaxAbsScaler()))
# steps.append(('classifier', MultinomialNB()))
# pipe = Pipeline(steps=steps, verbose=True)
# parameteres = {'imputer__strategy': ['mean'], 'imputer__fill_value': [2], 'classifier__alpha':[0, .5, 1]}
# grid_NB = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
#     n_splits=5, shuffle=True, random_state=0))
# grid_NB.fit(X_train, y_train)
# print("train score = %3.2f" % (grid_NB.score(X_train, y_train)))
# print("test score = %3.2f" % (grid_NB.score(X_test, y_test)))
# print(grid_NB.best_params_)

# %% =============================================================================
# Preprocessing
prep = []
prep.append(("imputer", SimpleImputer(
    missing_values=np.nan, strategy="constant", fill_value=2)))
prep.append(('scale', preprocessing.RobustScaler()))

# %% =============================================================================
# random forrest
steps = prep + [('classifier', RandomForestClassifier(random_state=0))]
pipe = Pipeline(steps=steps, verbose=True)
parameteres = {'imputer__strategy': ['constant'], 'imputer__fill_value': [2],
               'classifier__n_estimators': range(20, 200, 20),
               'classifier__criterion': ['gini', 'entropy', 'log_loss'],
               'classifier__min_samples_split': range(2, 10, 2),
               'classifier__min_samples_leaf': range(2, 10, 2),
               }
grid_RF = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
    n_splits=5, shuffle=True, random_state=0))
grid_RF.fit(X_train, y_train)
print("train score = %3.2f" % (grid_RF.score(X_train, y_train)))
print("test score = %3.2f" % (grid_RF.score(X_test, y_test)))
print(grid_RF.best_params_)
rf_pipe = grid_RF.best_estimator_

y_pred = grid_RF.predict(X_test)
report_grid_RF = classification_report_imbalanced(
    y_test, y_pred, target_names=['Focal', 'IGE'])

# %% =============================================================================
# gradient boosting
steps = prep + [('classifier', GradientBoostingClassifier(random_state=0))]
pipe = Pipeline(steps=steps, verbose=True)
parameteres = {'imputer__strategy': ['constant'], 'imputer__fill_value': [2],
               'classifier__n_estimators': range(20, 200, 20),
               'classifier__loss': ['log_loss', 'deviance', 'exponential'],
               'classifier__criterion': ['friedman_mse', 'squared_error', 'mse']
               }
grid_GB = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
    n_splits=5, shuffle=True, random_state=0))
grid_GB.fit(X_train, y_train)
print("train score = %3.2f" % (grid_GB.score(X_train, y_train)))
print("test score = %3.2f" % (grid_GB.score(X_test, y_test)))
print(grid_GB.best_params_)
gb_pipe = grid_GB.best_estimator_


y_pred = grid_GB.predict(X_test)
report_grid_GB = classification_report_imbalanced(
    y_test, y_pred, target_names=['Focal', 'IGE'])

# %% =============================================================================
# AdaBoost
steps = prep + [('classifier', AdaBoostClassifier(random_state=0))]
pipe = Pipeline(steps=steps, verbose=True)
parameteres = {'imputer__strategy': ['constant'], 'imputer__fill_value': [2],
               'classifier__n_estimators': range(10, 150, 20),
               'classifier__learning_rate': np.arange(.5, 3, .5),
               'classifier__algorithm': ['SAMME', 'SAMME.R'],
               }
grid_AB = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
    n_splits=5, shuffle=True, random_state=0))
grid_AB.fit(X_train, y_train)
print("train score = %3.2f" % (grid_AB.score(X_train, y_train)))
print("test score = %3.2f" % (grid_AB.score(X_test, y_test)))
print(grid_AB.best_params_)
ab_pipe = grid_AB.best_estimator_



y_pred = grid_AB.predict(X_test)
report_grid_AB = classification_report_imbalanced(
    y_test, y_pred, target_names=['Focal', 'IGE'])

# %% =============================================================================
# XBoost
# steps = prep + [('classifier', XGBClassifier(objective='binary:logistic',
#                                              booster='gbtree',
#                                              eval_metric='auc',
#                                              tree_method='hist',
#                                              grow_policy='lossguide',
#                                              use_label_encoder=False))]

# pipe = Pipeline(steps=steps, verbose=True)
# parameteres = {'imputer__strategy': ['constant'], 'imputer__fill_value': [2],
#                'classifier__min_child_weight': [1, 5, 10],
#                'classifier__gamma': [0.5, 1, 1.5, 2, 5],
#                'classifier__subsample': [0.6, 0.8, 1.0],
#                'classifier__colsample_bytree': [0.6, 0.8, 1.0],
#                'classifier__max_depth': [3, 4, 5],
#                }

# grid_XGB = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
#     n_splits=5, shuffle=True, random_state=0))
# grid_XGB.fit(X_train, y_train)
# print("train score = %3.2f" % (grid_XGB.score(X_train, y_train)))
# print("test score = %3.2f" % (grid_XGB.score(X_test, y_test)))
# print(grid_XGB.best_params_)
# xgb_pipe = grid_XGB.best_estimator_


# %% =============================================================================
# Bagging
steps = prep + [('classifier', BaggingClassifier(random_state=0))]
pipe = Pipeline(steps=steps, verbose=True)
parameteres = {'imputer__strategy': ['constant'], 'imputer__fill_value': [2],
               'classifier__n_estimators': [5, 10, 15, 20, 30, 40, 50, 60, 80, 100],
               'classifier__max_samples': np.arange(.05, .8, .1),
               'classifier__max_features': np.arange(.05, 1, .1),
               }
grid_Bag = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
    n_splits=5, shuffle=True, random_state=0))
grid_Bag.fit(X_train, y_train)
print("train score = %3.2f" % (grid_Bag.score(X_train, y_train)))
print("test score = %3.2f" % (grid_Bag.score(X_test, y_test)))
print(grid_Bag.best_params_)
bag_pipe = grid_Bag.best_estimator_



y_pred = grid_Bag.predict(X_test)
report_grid_Bag = classification_report_imbalanced(
    y_test, y_pred, target_names=['Focal', 'IGE'])

# %% =============================================================================
# ExT
steps = prep + [('classifier', ExtraTreesClassifier(random_state=0))]
pipe = Pipeline(steps=steps, verbose=True)
parameteres = {'imputer__strategy': ['constant'], 'imputer__fill_value': [2],
               'classifier__n_estimators': [10, 20, 30, 50, 70, 100, 150, 200],
               'classifier__criterion': ['gini', 'entropy', 'log_loss'],
               'classifier__min_samples_split': range(1, 10),
               'classifier__min_samples_leaf': range(1, 10),
               }
grid_ExT = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
    n_splits=5, shuffle=True, random_state=0))
grid_ExT.fit(X_train, y_train)
print("train score = %3.2f" % (grid_ExT.score(X_train, y_train)))
print("test score = %3.2f" % (grid_ExT.score(X_test, y_test)))
print(grid_ExT.best_params_)
ext_pipe = grid_ExT.best_estimator_


y_pred = grid_ExT.predict(X_test)
report_grid_ExT = classification_report_imbalanced(
    y_test, y_pred, target_names=['Focal', 'IGE'])

# %% =============================================================================
# StC
# steps = []
# steps.append(("imputer", SimpleImputer(
#     missing_values=np.nan, strategy="constant", fill_value=2)))
# steps.append(('scale', preprocessing.MaxAbsScaler()))
# steps.append(('classifier', StackingClassifier()))
# pipe = Pipeline(steps=steps, verbose=True)
# parameteres = {'imputer__strategy': ['constant'], 'imputer__fill_value': [2]
#                }
# grid_StC = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
#     n_splits=5, shuffle=True, random_state=0))
# grid_StC.fit(X_train, y_train)
# print("train score = %3.2f" % (grid_StC.score(X_train, y_train)))
# print("test score = %3.2f" % (grid_StC.score(X_test, y_test)))
# print(grid_StC.best_params_)

# %% =============================================================================
# SVM
steps = prep + [('classifier', SVC(random_state=0))]
pipe = Pipeline(steps=steps, verbose=True)
parameteres = {'imputer__strategy': ['constant'], 'imputer__fill_value': [2],
               'classifier__C': np.arange(0.5, 5, .5),
               'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
               'classifier__gamma': ['scale', 'auto'],
               }
grid_SVM = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
    n_splits=5, shuffle=True, random_state=0))
grid_SVM.fit(X_train, y_train)
print("train score = %3.2f" % (grid_SVM.score(X_train, y_train)))
print("test score = %3.2f" % (grid_SVM.score(X_test, y_test)))
print(grid_SVM.best_params_)
svm_pipe = grid_SVM.best_estimator_


y_pred = grid_SVM.predict(X_test)
report_grid_SVM = classification_report_imbalanced(
    y_test, y_pred, target_names=['Focal', 'IGE'])



# %% =============================================================================
# LinearSVM
# steps = prep + [('classifier', LinearSVC(random_state=0))]
# pipe = Pipeline(steps=steps, verbose=True)
# parameteres = {'imputer__strategy': ['constant'], 'imputer__fill_value': [2],
#                'classifier__C': np.arange(0.5,5,.5), 'classifier__penalty': ['l1', 'l2'],
#                'classifier__loss': ['hinge', 'squared_hinge']}
# grid_LSVM = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
#     n_splits=5, shuffle=True, random_state=0))
# grid_LSVM.fit(X_train, y_train)
# print("train score = %3.2f" % (grid_LSVM.score(X_train, y_train)))
# print("test score = %3.2f" % (grid_LSVM.score(X_test, y_test)))
# print(grid_LSVM.best_params_)
# lsvm_pipe = grid_LSVM.best_estimator_

# %% =============================================================================
# logistic regression
steps = prep + [('classifier', LogisticRegression(random_state=0))]
pipe = Pipeline(steps=steps, verbose=True)
parameteres = {'imputer__strategy': ['constant'], 'imputer__fill_value': [2],
               'classifier__C': np.arange(0.5, 5, .5), 'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],
               'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
grid_LR = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
    n_splits=5, shuffle=True, random_state=0))
grid_LR.fit(X_train, y_train)
print("train score = %3.2f" % (grid_LR.score(X_train, y_train)))
print("test score = %3.2f" % (grid_LR.score(X_test, y_test)))
print(grid_LR.best_params_)
lr_pipe = grid_LR.best_estimator_


y_pred = grid_LR.predict(X_test)
report_grid_LR = classification_report_imbalanced(
    y_test, y_pred, target_names=['Focal', 'IGE'])


# %% =============================================================================
# KNN
steps = prep + [('classifier', KNeighborsClassifier())]
pipe = Pipeline(steps=steps, verbose=False)
parameteres = {'imputer__strategy': ['constant'], 'imputer__fill_value': [2],
               'classifier__weights': ['uniform', 'distance'],
               'classifier__n_neighbors': [3, 13, 23, 33],
               'classifier__leaf_size': range(10, 100, 10),
               'classifier__p': [1, 2]}
grid_KNN = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
    n_splits=5, shuffle=True, random_state=0))
grid_KNN.fit(X_train, y_train)
print("train score = %3.2f" % (grid_KNN.score(X_train, y_train)))
print("test score = %3.2f" % (grid_KNN.score(X_test, y_test)))
print(grid_KNN.best_params_)
knn_pipe = grid_KNN.best_estimator_


y_pred = grid_KNN.predict(X_test)
report_grid_KNN = classification_report_imbalanced(
    y_test, y_pred, target_names=['Focal', 'IGE'])


# %% =============================================================================
# DTree
steps = prep + [('classifier', DecisionTreeClassifier(random_state=0))]
pipe = Pipeline(steps=steps, verbose=False)
parameteres = {'imputer__strategy': ['constant'], 'imputer__fill_value': [2],
               'classifier__criterion': ['gini', 'entropy', 'log_loss'],
               'classifier__min_samples_split': range(1, 10),
               'classifier__min_samples_leaf': range(1, 10),
               'classifier__max_features': np.arange(.05, 1, .1)
               }
grid_DT = model_selection.GridSearchCV(pipe, param_grid=parameteres, cv=model_selection.StratifiedKFold(
    n_splits=5, shuffle=True, random_state=0))
grid_DT.fit(X_train, y_train)
print("train score = %3.2f" % (grid_DT.score(X_train, y_train)))
print("test score = %3.2f" % (grid_DT.score(X_test, y_test)))
print(grid_DT.best_params_)
dt_pipe = grid_DT.best_estimator_

# %% =============================================================================
# Voting ...
estimators = [
    ('rf', rf_pipe),
    ('gb', gb_pipe),
    ('ab', ab_pipe),
    ('bag', bag_pipe),
    ('ext', ext_pipe),
    ('svm', svm_pipe),
    # ('lsvm', grid_LSVM),
    ('lr', lr_pipe),
    ('knn', knn_pipe),
    # ('dt', dt_pipe)
]

VotClf = VotingClassifier(estimators=estimators, voting='hard')
VotClf.fit(X_train, y_train)
print("train score = %3.3f" % (VotClf.score(X_train, y_train)))
print("test score = %3.3f" % (VotClf.score(X_test, y_test)))


# %% manual voting
# def voting(df):
#     a = []
#     for i in range(df.shape[0]):
#         a.append(df.iloc[i, :].mode()[0])
#     return pd.DataFrame(a)


# y = pd.DataFrame({'RF': grid_RF.predict(X_test), 'SVM': grid_SVM.predict(X_test), 'LSVM': grid_LSVM.predict(
#     X_test), 'knn': grid_KNN.predict(X_test), 'LR': grid_KNN.predict(X_test)})
# y = voting(y)

# accuracy_score(y_test, y)

# %% =============================================================================
# Stacking ...
estimators = [
    ('rf', rf_pipe),
    ('gb', gb_pipe),
    ('ab', ab_pipe),
    ('bag', bag_pipe),
    ('ext', ext_pipe),
    ('svm', svm_pipe),
    # ('lsvm', lsvm_pipe),
    ('lr', lr_pipe),
    ('knn', knn_pipe),
    # ('dt', dt_pipe)
]

StClf = StackingClassifier(estimators=estimators,
                           cv=model_selection.StratifiedKFold(n_splits=5),
                           stack_method='predict',
                           # 'decision_function',
                           # 'predict',
                           # 'predict_proba'
                           )
StClf.fit(X_train, y_train)
print("train score = %3.3f" % (StClf.score(X_train, y_train)))
print("test score = %3.3f" % (StClf.score(X_test, y_test)))

ConfusionMatrixDisplay.from_estimator(StClf, X_test, y_test)
plt.show()


def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report_r3.csv', index=False)


y_pred = StClf.predict(X_test)
report = classification_report_imbalanced(
    y_test, y_pred, target_names=['Focal', 'IGE'])
# classification_report_csv(report)
# %% =============================================================================
# worst case
# accuracy_score(y_test, np.full((521,1),'Focal'))


# %% =============================================================================
# # save the results

# pickle
def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True


fileName = 'p2_resultSave_30prc_test_r3.pkl'

bk = {}
for k in dir():
    obj = globals()[k]
    if is_picklable(obj):
        try:
            bk.update({k: obj})
        except TypeError:
            pass

# to save session
with open('./' + fileName, 'wb') as f:
    pickle.dump(bk, f)

# to load your session
with open('./' + fileName, 'rb') as f:
    bk_restore = pickle.load(f)
for k in bk_restore:
    globals()[k] = bk_restore[k]
bk_restore.close()

# %%
# other save methods





# shelve
# import shelve
# bk = shelve.open(fileName,'n')
# for k in dir():
#     try:
#         bk[k] = globals()[k]
#     except Exception:
#         pass
# bk.close()
# # to restore
# bk_restore = shelve.open(fileName)
# for k in bk_restore:
#     globals()[k] = bk_restore[k]
# bk_restore.close()




# dill
# fileName = 'resultSave_30prc_test.pkl'
# f = open(fileName, 'wb')
# dill.dump_session(fileName)
# f.close()
# print(fileName+' is saved!')

# dill.load_session('./' + fileName)


# %% save classifier model
fileName = 'classifierModel.pkl'
with open('./' + fileName, 'wb') as f:
    pickle.dump(StClf, f)
    
fileName = 'inputEncoder.pkl'
with open('./' + fileName, 'wb') as f:
    pickle.dump(enc, f)

    



# %% Statistical test

from scipy.stats import ttest_ind, chi2_contingency


# Split numerical and categorical columns
numerical_cols = [0, 6]  # Indices of numerical columns in X
categorical_cols = [1, 2, 3, 4, 5, 7, 8]  # Indices of categorical columns in X


table = pd.DataFrame(columns=['Feature', 'Test', 'p-value'])

# Perform t-tests on numerical columns
for col in numerical_cols:
    feature_values = X_train[:, col]
    class_0_values = feature_values[y_train == 'Focal']
    class_1_values = feature_values[y_train == 'IGE']
    _, p_value = ttest_ind(class_0_values, class_1_values)
    table = table.append({'Feature': feature_names[col], 'Test': 't-test', 'p-value': p_value}, ignore_index=True)

# Perform chi-square tests on categorical columns
for col in categorical_cols:
    uniqVal = np.unique(X_train[:, col]).flatten()
    uniqVal = uniqVal[~np.isnan(uniqVal)]
    contingency_table = np.array([[sum((X_train[:, col] == value) & (y_train == 'Focal')) for value in uniqVal],
                                  [sum((X_train[:, col] == value) & (y_train == 'IGE')) for value in uniqVal]])
    _, p_value, _, _ = chi2_contingency(contingency_table)
    table = table.append({'Feature': feature_names[col], 'Test': 'Chi-square', 'p-value': p_value}, ignore_index=True)


# Display the table
print(table)












