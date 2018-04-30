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
from xgboost import XGBClassifier

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) 

#defining RobustScaler for the nuerical and onehot for categorical features to be used in the pipeline
scaler_encoder = ScalerAndOneHotEncoder(
    RobustScaler(), categorical_index=categorical_index)

#############################################################################


##################### Random Forest Classifier ###########################
#Preparing the pipeline
steps = [('scaler_encoder', scaler_encoder),
        ('XGB', XGBClassifier() )]
pipeline = Pipeline(steps)

# Meta parameters
n_estimators = [300,400, 600, 800, 900]
booster = ['gbtree',  'dart']
subsample = [0.5, 0.65, 0.8, 1]
scale_pos_weight = [len(y[y==False])/len(y), 1 ]
max_depth = [3,5, 6, 10, 12]
gamma = [i/10.0 for i in range(0,5)]
eta = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]
XGB_estimator = GridSearchCV(pipeline,
                         dict(
                             XGB__n_estimators=n_estimators,
                             XGB__max_depth=max_depth,
                             XGB__subsample=subsample,
                             XGB__booster=booster,
                             XGB__scale_pos_weight=scale_pos_weight,
                             XGB__gamma=gamma,
                             XGB__learning_rate=eta),
                        cv=5,
                        scoring='roc_auc',
                        verbose=3,
                        n_jobs=24)
                         
XGB_estimator.fit(X_train, np.ravel(y_train))
print (XGB_estimator.best_score_) 
print (XGB_estimator.best_params_)
y_pred = XGB_estimator.predict(X_test)
print(met.confusion_matrix(y_test,y_pred, labels=[0,1]))
print(met.f1_score(y_test,y_pred))
pickle.dump(XGB_estimator, 
            open( savedir+'XGB_Estimator.p', 'wb' ) )

####################3 Saving the results #####################################
best_params = {'XGB': 
               {'Parameters': XGB_estimator.best_params_ ,
                'Best Score': XGB_estimator.best_score_}}
             
pickle.dump( best_params, open( 'best_params_XGB_auc.p', 'wb' ) )
