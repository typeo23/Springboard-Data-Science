#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A utility functions to load filter and manipulate the dataset
Created on Mon Apr 23 14:10:27 2018

@author: itay
"""

import pandas as pd

def filter_extreme_earns(df, cutoff=4):
    """
    A utility function to filter from df people who earn more then cutoff*Sigma
    of thire occupation mean salary
    """
    occupations = df.peioocc.unique()

    #Dictionary complerhansion for the cutoff salary for each occupation
    occ_cutoff = {peioocc : df[df.peioocc == peioocc].ern_val.mean()
                          +cutoff*df[df.peioocc == peioocc].ern_val.std()
                  for peioocc in  occupations}
    
    #Creating a DataSeries with the the cutoff salary for each individual
    filt_col = df.peioocc.map(occ_cutoff)
    #filtering the outliers
    filt_df = df[df.ern_val < filt_col]
    return filt_df

def load_filter_dataset(filename, categorical_features, numerical_features,
                        labels, threshold, index_col='peridnum',
                        max_ern=1e9, min_ern=0, filt_sigma=3,
                        ):
   """
   Loads the datase filter it and create a mtrix of features and lables
   
   
   Parameters:
   -----------
   filename   : A .csv.bz file to load
   max_ern    : Individuals earning more then this value will be filtered out
   min_ern    :  Individuals earning less then this value will be filtered out
   filt_sigma : For each occupation will filter people earning more then
                filt_sigma * mean for their occupation
   categorical_features   : List of categorical 
                            fetures to pass to the classifier
  numerical_features   : List of numerical 
                            fetures to pass to the classifier
   label      : list containing the label colums
   threshold  : Threshold value for binarizing the labels
  """   
    
    
   df = pd.read_csv(
   filename, compression='bz2',
   index_col=index_col)

   #prediction label
   features = categorical_features + numerical_features
   # filtering income, and peopel makeing more then 3\sigma the mean 
   # salary for their occupation 
   df_filt = df[(df['ern_val'] > min_ern) & (df['ern_val'] < max_ern)]
   df_filt = filter_extreme_earns(df_filt, filt_sigma)

   # converting colum names to uppercase
   df_filt.columns = df_filt.columns.str.upper()
   df_filt = df_filt[features + labels]
   # Getting the index of categorical variables for the OneHot encoder.
   categorical_index = [
       i for i, x in enumerate(df_filt.columns.values)
       if x in categorical_features
   ]
    
    
   #############################################################################
    
   # getting features
   X = df_filt[features].values
   # Setting a binary tager from the ern_val column
   y = (df_filt[labels] > threshold).values
    
   return X, y, categorical_index