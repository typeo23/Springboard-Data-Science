import pandas as pd
import sys
import csv
from polyglot.detect import Detector
import re

maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

df = pd.read_csv(open('../data/csv/s2-corpus-00test2.csv', 'rU'), engine='c')
lang = []
for i, abstract in enumerate(df.paperAbstract.values):
    try:
        lang.append(Detector(abstract, quiet=True).language.name)
    except:
        lang.append('unKnown')
df['lang'] = lang
df.to_csv('../data/csv/s2-corpus-002_lang.csv')