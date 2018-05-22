import pandas as pd
from gensim.models.doc2vec import TaggedDocument


def return_labeled_docs_from_df(df):
    labels = df.id.values
    docs = df.paperAbstract.values

    return labels, docs


class LabeledLineSentence(object):
    def __init__ (self, doc_list, labels_list):
        self.doc_list = doc_list
        self.labels_list = labels_list

    def __iter__(self):
        for idx, doc in zip(self.labels_list, self.doc_list):
            yield TaggedDocument(words=doc.split(), tags=[idx])

