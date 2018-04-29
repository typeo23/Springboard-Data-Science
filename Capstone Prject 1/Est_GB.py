# -*- coding: utf-8 -*-
"""
Grid Searchin to evaluate for GB Classifier
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ScalerAndOneHotEncoder import ScalerAndOneHotEncoder
from sklearn.preprocessing import StandardScaler
from Filter_extreme_earns import filter_extreme_earns
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

import sklearn.metrics as met

from dataset_manipulation_funcs import load_filter_dataset

import pickle

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
            'Data/income_data_2017_clean_zeros.csv.bz2',
            max_ern=250000, min_ern=1000,
            categorical_features=categorical_features, 
            numerical_features=numerical_features,
            labels=labels, threshold=40000,
            filt_sigma=2)

# Spliting to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) 

#defining RobustScaler for the nuerical and onehot for categorical features to be used in the pipeline
scaler_encoder = ScalerAndOneHotEncoder(
    RobustScaler(), categorical_index=categorical_index)

#############################################################################


##################### GB Classifier ###########################
#Preparing the pipeline
steps = [('scaler_encoder', scaler_encoder),
        ('GB', GradientBoostingClassifier() )]
pipeline = Pipeline(steps)

# Meta parameters
loss = ['deviance', 'exponential']
n_estimators = [600, 700, 800, 900]
subsample = [0.3, 0.5, 0.65, 0.8, 1]
max_features = ['sqrt', 'log2', 0.4]
max_depth = [None, 3, 5]
min_samples_leaf = [1, 2, 4]
min_samples_split = [2, 3]

GB_estimator = GridSearchCV(pipeline,
                         dict(
                             GB__loss=loss,
                             GB__n_estimators=n_estimators,
                             GB__subsample=subsample,
                             GB__max_features=max_features,
                             GB__min_samples_leaf=min_samples_leaf,
                             GB__min_samples_split=min_samples_split),
                        cv=5,
                        scoring='roc_auc',
                        verbose=3,
                        n_jobs=18)
                         
GB_estimator.fit(X_train, np.ravel(y_train))
print (GB_estimator.best_score_) 
print (GB_estimator.best_params_)
y_pred = GB_estimator.predict(X_test)
print(met.confusion_matrix(y_test,y_pred, labels=[0,1]))
print(met.f1_score(y_test,y_pred))
pickle.dump( GB_estimator, 
            open( savedir+'GB_Estimator_auc.p', 'wb' ) )

####################3 Saving the results #####################################
best_params = {'Gradient Boost': 
               {'Parameters': GB_estimator.best_params_ ,
                'Best Score': GB_estimator.best_score_}}
             
pickle.dump( best_params, open( 'best_params_GB_auc.p', 'wb' ) )
