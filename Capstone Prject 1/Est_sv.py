# -*- coding: utf-8 -*-
"""
Grdi Searchin to evaluate different 
classifiers on the dataset
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as met

from dataset_manipulation_funcs import load_filter_dataset

import pickle

##############################################################################
savedir = 'best_estimators/'
# Import the dataset
df = pd.read_csv(
    'Data/income_data_2017_clean_zeros.csv.bz2',
    compression='bz2',
    index_col='peridnum')

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
X, y, categorical_index = load_filter_dataset(
        'Data/income_data_2017_clean_zeros.csv.bz2',
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

##### Starting with Logisting Regression Classfier ##########################
steps = [('scaler_encoder', scaler_encoder),
        ('logistic', LogisticRegression() )]
pipeline = Pipeline(steps)


# Meta parameters
cs = np.logspace(-2, 2, 4)
penalty = ['l2', 'l1']
class_weight = [None, 'balanced']

logistic_estimator = GridSearchCV(pipeline,
                         dict(
                             logistic__C=cs,
                             logistic__penalty=penalty,
                             logistic__class_weight=class_weight),
                        cv=5,
                        scoring='f1',
                        n_jobs=20,
                        verbose=5)
                         
logistic_estimator.fit(X_train, y_train)
print (logistic_estimator.best_score_) 
print (logistic_estimator.best_params_)         


y_pred = logistic_estimator.predict(X_test)
print(met.confusion_matrix(y_test,y_pred, labels=[0,1]))
print(met.f1_score(y_test,y_pred))         
pickle.dump( logistic_estimator, 
            open( savedir+'Logisitic_Estimator.p', 'wb' ) )
##################### Random Forest Classifier ###########################
#Preparing the pipeline
steps = [('scaler_encoder', scaler_encoder),
        ('rf', RandomForestClassifier(n_jobs=20) )]
pipeline = Pipeline(steps)

# Meta parameters
n_estimators = [200, 300, 400, 500]
criterion = ['gini', 'entropy']
max_features = ['sqrt', 'log2']
max_depth = [None, 3, 5]
min_samples_leaf = [1, 2, 4]
min_samples_split = [2, 3]

rf_estimator = GridSearchCV(pipeline,
                         dict(
                             rf__n_estimators=n_estimators,
                             rf__criterion=criterion,
                             rf__max_features=max_features,
                             rf__min_samples_leaf=min_samples_leaf,
                             rf__min_samples_split=min_samples_split),
                        cv=5,
                        scoring='f1',
                        verbose=5)
                         
rf_estimator.fit(X_train, y_train)
print (rf_estimator.best_score_) 
print (rf_estimator.best_params_)
y_pred = rf_estimator.predict(X_test)
print(met.confusion_matrix(y_test,y_pred, labels=[0,1]))
print(met.f1_score(y_test,y_pred))
pickle.dump( rf_estimator, 
            open( savedir+'RandomForest_Estimator.p', 'wb' ) )
################################### SVC Classifier ##########################
#Preparing the pipeline
steps = [('scaler_encoder', scaler_encoder),
        ('svc', SVC() )]
pipeline = Pipeline(steps)
# Meta parameters
C = [0.1, 1, 5, 10, 30]
gamma =  ['auto', 1e-2, 1e-3]


svc_estimator = GridSearchCV(pipeline,
                         dict(
                             svc__C=C,
                             svc__gamma=gamma
                             ),
                        cv=5,
                        scoring='f1',
                        n_jobs=20,
                        verbose=5)
                         
svc_estimator.fit(X_train, y_train)
print (rf_estimator.best_score_) 
print (rf_estimator.best_params_)
y_pred = rf_estimator.predict(X_test)
print(met.confusion_matrix(y_test,y_pred, labels=[0,1]))
print(met.f1_score(y_test,y_pred))
pickle.dump( svc_estimator, 
            open( savedir+'SVM_Estimator.p', 'wb' ) )
####################3 Saving the results #####################################
best_params = {'Logistic Regression': 
               {'Parameters': logistic_estimator.best_params_ ,
                'Best Score': logistic_estimator.best_score_},
              'Random Forest': 
               {'Parameters': rf_estimator.best_params_ ,
                'Best Score': rf_estimator.best_score_},
              'SVM Classifier':
              {'Parameters': svc_estimator.best_params_ ,
                'Best Score': svc_estimator.best_score_}
              }

pickle.dump( best_params, open( 'best_params_Log_Forest_SVM_2.p', 'wb' ) )