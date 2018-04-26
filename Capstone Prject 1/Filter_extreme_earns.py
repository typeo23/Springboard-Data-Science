#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A utility function to filter people who earn more then cutoff*Sigma
of thire occupation mean salary
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
    ouliter_series = filt_col = df.peioocc.map(occ_cutoff)
    #filtering the outliers
    filt_df = df[df.ern_val < filt_col]
    return filt_df