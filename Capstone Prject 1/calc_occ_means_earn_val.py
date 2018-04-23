# -*- coding: utf-8 -*-
"""
A simple script to create the average and st salary for each occupation
The output is needed for the filtering step in the pipeline
"""

import numpy as np
import pandas as pd
import pickle

data_dir = 'Data/'
filename = 'income_data_2017_clean_zeros.csv.bz2'
df= pd.read_csv(data_dir+filename, compression='bz2', index_col='peridnum')

occupations = df.peioocc.unique()

#Dictionary complerhansion for the mean and std for each occupation
occ_means_stds = {peioocc : {'mean': df[df.peioocc == peioocc].ern_val.mean(),
                              'std': df[df.peioocc == peioocc].ern_val.std(),
                              'std4':df[df.peioocc == peioocc].ern_val.mean()
                                    +4*df[df.peioocc == peioocc].ern_val.std()}
                          for peioocc in  occupations}
pickle.dump(occ_means_stds, open(data_dir+'occ_means_stds.p', 'wb'))