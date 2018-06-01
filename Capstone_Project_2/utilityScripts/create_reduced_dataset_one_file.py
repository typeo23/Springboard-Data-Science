"""
A script to filter one file of the dataset
removes all columns except document id , abstract journal and authors
The result is saved to a compressed csv file
"""

import pandas as pd


def meaningful_lines(x, line_length, paragraph_length):
    """
    Utility function which checks if the abstract is linger then @paragraph_length sentances
    A line is considered a senance if its linget then @line_length
    :param x: Input paragraph
    :param line_length: Minimum number of words in a line to be considered a sentance
    :param paragraph_length: Minimum paragraph_length
    :return: True if the paragraph contains more then @paragraph_length sentances
    """
    lines = x.split('.')
    lines = [line for line in lines if len(line.split(' ')) > line_length]
    return len(lines) > paragraph_length


data_dir = '../data/'
filename = 's2-corpus-00.gz'
out_dir = '../data/csv/'
out_filename = filename.split('.')[0]+'test2'+'.csv'
columns_to_keep = ['id',  'paperAbstract', 'title', 'year', 'journalName']

reader = pd.read_json(data_dir+filename, lines=True, chunksize=1e5)

# header only for the first chunk
first = True
for df in reader:
    df = df[columns_to_keep]
    df = df[df.journalName != '']
    df = df[df.paperAbstract.map(lambda x: meaningful_lines(x, 4, 6))]
    df.to_csv(out_dir+out_filename, index=False, header=first, mode='a')
    first = False



