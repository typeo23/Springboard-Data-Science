# -*- coding: utf-8 -*-
"""
Grid Searchin to evaluate For XGBoost Calssifier
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ScalerAndOneHotEncoder import ScalerAndOneHotEncoder
from sklearn.preprocessing import StandardScaler
from Filter_extreme_earns import filter_extreme_earns
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import lightgbm as lgb

import sklearn.metrics as met

from dataset_manipulation_funcs import load_filter_dataset

import pickle

import warnings

warnings.simplefilter('ignore', DeprecationWarning)
##############################################################################
savedir = 'best_estimators/'
# Import the dataset

categorical_features = [
    'PEIOOCC', 'A_HGA', 'PRDTRACE', 'FKIND', 'A_CLSWKR', 'A_WKSTAT', 'A_MJIND',
    'MJOCC', 'PTYN', 'LJCW', 'A_SEX', 'WEMOCG', 'MIG_CBST', 'MIGSAME',
    'H_TYPE', 'H_LIVQRT', 'GTCBSA', 'GESTFIPS'
]
numerical_features = [
    'H_NUMPER', 'FPERSONS', 'FOWNU6', 'FOWNU18', 'A_AGE', 'A_HRS1', 'A_USLHRS',
    'PHMEMPRS', 'HRSWK', 'HNUMFAM'
]
#prediction label
labels = ['ERN_VAL']
features = categorical_features + numerical_features
#Dont load the dataset again if it is already loaded (for debuging purpose)
if 'X' not in locals():
    X, y, categorical_index = load_filter_dataset(
            'data/income_data_2017_clean_zeros.csv.bz2',
            max_ern=250000, min_ern=5000,
            categorical_features=categorical_features, 
            numerical_features=numerical_features,
            labels=labels, threshold=40000)

# Spliting to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.55) 

estimator = lgb.LGBMClassifier(nthread=4,silent=False)
param_grid = {
    'learning_rate': [0.01, 0.05,0.04],
    'n_estimators': [10],
    'max_depth': [-1, 3, 6],
    'boosting_type' : ['dart', 'gbdt'],
    'objective' : ['binary'],
    'seed' : [777],
    'colsample_bytree' : [0.7,1],
    'subsample' : [0.8,1],
    'reg_alpha' : [0,0.1,1],
    'reg_lambda' : [0,1,2,],
    'is_unbalance' : [True,False],
    'categorical_features': [categorical_index],
    "boost_from_average": [True, False],
    'feature_fraction': [0.7, 0.8, 1]
}

gbm_estimator = GridSearchCV(estimator, param_grid,
                                   n_jobs=5,cv=5,
                                   verbose=1)
                         
gbm_estimator.fit(X_train, np.ravel(y_train))
print (gbm_estimator.best_score_) 
print (gbm_estimator.best_params_)
y_pred = gbm_estimator.predict(X_test)
print(met.confusion_matrix(y_test,y_pred, labels=[0,1]))
print(met.f1_score(y_test,y_pred))
pickle.dump(gbm_estimator, 
            open( savedir+'gbm_estimator_acc.p', 'wb' ) )

####################3 Saving the results #####################################
best_params = {'gbm_estimator': 
               {'Parameters': gbm_estimator.best_params_ ,
                'Best Score': gbm_estimator.best_score_}}
             
pickle.dump( best_params, open( 'best_params_gbm_estimator_acc.p', 'wb' ) )
preds = gbm_estimator.predict_proba(X_test)[:,1]

fpr, tpr, _ = met.roc_curve(y_test, preds)

plt.figure()
lw = 2
auc = met.auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = {:0.2f})'.format(auc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
