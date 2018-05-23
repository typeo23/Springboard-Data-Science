import os
import pandas as pd
from trainText.trainingFunctions import return_labeled_docs_from_df
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import TaggedDocument

class LabeledDir(object):
    def __init__(self, files_dir='', file_ext=''):
        self.files_dir = files_dir
        self.file_ext = file_ext

    def __iter__(self):
        files = [f for f in os.listdir(self.files_dir) if f.split('.')[1] == self.file_ext]
        for file in files:
            reader = pd.read_json(os.path.join(self.files_dir, file), lines=True, chunksize=100)
            for df in reader:
                labels, docs = return_labeled_docs_from_df(df)
                tokenizer = RegexpTokenizer(r'\w+')
                for idx, doc in zip(labels, docs):
                    yield TaggedDocument(words=tokenizer.tokenize(doc), tags=[idx])