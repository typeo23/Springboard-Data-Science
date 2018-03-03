"""
First cleaning of the  dataset
removing some of theirrelevant columns (and colums that have more then 70% missing data)
and filtering children and unemployed 
"""

import numpy as np
import pandas as pd

# Loading the dataset all negative numbers  are NaN 
df= pd.read_csv('21321-0001-Data.tsv', delimiter='\t', index_col ='peridnum')
df[df <0] = np.nan
#list of uncessery columns, see pdf code manual
drop_cols = ['H_RESPNM', 'H_YEAR', 'H_HHTYPE', 'H_MONTH', 'H_MIS', 'H_TYPEBC', 'H_TELAVL', 'H_TELINT', 'GTINDVPC',
              'GTCSA', 'HCHINNO', 'HHOTLUN', 'HHOTNO', 'HFLUNCH', 'HFLUNNO', 'HFDVAL', 'HENGVAL', 'HSUR_YN', 'HSURVAL',
              'HRNTVAL', 'HFINVAL', 'FOIVAL', 'F_MV_FS', 'F_MV_SL', 'FFNGCARE', 'FFNGCAID', 'FHOUSSUB', 'FFOODREQ',
              'FHOUSREQ', 'PEAFWHN1', 'PEAFWHN2', 'PEAFWHN3', 'PEAFWHN4', 'SUR_SC2', 'DIS_VAL2', 'PEABSRSN', 'OTHSTYP2',
              'OTHSTYP3', 'AHITYP1', 'AHITYP2', 'OTHSTYP4', 'OTHSTYP5', 'OTHSTYP6', 'AHITYP3', 'AHITYP4', 'AHITYP5',
              'AHITYP6']

# converting to lowercase and dropping the cols, to save some disk space
drop_cols = [col.lower() for col in drop_cols]
df = df.drop(drop_cols, axis=1)

# Filtering unemployed
df = df[df['peioind'] >0]

#filtering age
df = df[(df['a_age'] > 17) & (df['a_age'] < 75)]


# Saving csv and compressing to save diskspace
df.to_csv('./data/income_data_2017_clean_zeros.csv.bz2', compression='bz2')