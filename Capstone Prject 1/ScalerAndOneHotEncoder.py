"""
A calss for scaling numerical features while one hot encoding 
categorical features
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.sparse import coo_matrix, hstack

class ScalerAndOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    A calss for scaling numerical features while one hot encoding 
    categorical features
    """
    def __init__(self, scaler, categorical_index, numerical_index=None):
        """
        scaler is a numerical Scaler class, i.e StandardScaler ect.
        categorical_index: a list of the categorical features indices
        numerical_index: numerical features indices, if none will be
                         deduced from the size of the input
        """
        self.scaler = scaler
        self.encoder = OneHotEncoder()
        self.categorical_index = categorical_index
        self.numerical_index = numerical_index
    
    def fit(self, X, y=None):
        """
        fitting the scaler and encoder. If the numerical indices are missing 
        get them from the shap of X
        """
        n = X.shape[1]
        if self.numerical_index == None:
            self.numerical_index = [x for x in range(n) 
                                    if x not in self.categorical_index]
        self.encoder.fit(X[:, self.categorical_index], y)
        self.scaler.fit(X[:, self.numerical_index], y)
        return self
        
    def transform(self, X):
        numerical_transform =  self.scaler.transform(X[:, self.numerical_index])
        categorical_transfrom = self.encoder.transform(X[:, self.categorical_index])
        sprse_concat = hstack([coo_matrix(numerical_transform),
                           categorical_transfrom])
        return sprse_concat
        