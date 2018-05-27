"""
A script to filter one file of the dataset
removes all columns except document id , abstract journal and authors
The result is saved to a compressed csv file
"""

import pandas as pd

data_dir = '../data/'
filename = 's2-corpus-00.gz'
out_dir = '../data/csv/'
out_filename = filename.split('.')[0]+'.csv'
columns_to_keep = ['id',  'paperAbstract', 'title', 'year', 'journalName']

reader = pd.read_json(data_dir+filename, lines=True, chunksize=1e5)

# header only for the first chunk
first = True
for df in reader:
    df = df[columns_to_keep]
    df = df[df.paperAbstract.map(lambda x: len(x.split('.')) > 5)]
    df.to_csv(out_dir+out_filename, index=False, header=first, mode='a')
    first = False



