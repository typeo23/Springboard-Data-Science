import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import RegexpTokenizer


def return_labeled_docs_from_df(df):
    """
    Takes a pandas dataframe produced from reading the json files
    obtained from semantic scholar dataset and produce labels and documents
    to create a labeled doc from. The function removes papers which have null abstract
    :param df: dataframe
    :return: labels: list of paper id's, docs: list of the corresponding abstracts
    """
    df_filt = df[df.paperAbstract != '']
    labels = df_filt.id.values
    docs = df_filt.paperAbstract.values

    return labels, docs


class LabeledLineSentence(object):
    """
    Takes a dic list and label list (returned from @return_labeled_docs_from_df()
    and return an iterator over labeled documents. Tokanization is done by simply splitting at ' '
    """
    def __init__ (self, doc_list, labels_list):
        self.doc_list = doc_list
        self.labels_list = labels_list

    def __iter__(self):
        for idx, doc in zip(self.labels_list, self.doc_list):
            yield TaggedDocument(words=doc.split(), tags=[idx])


class LabeledLineSentenceRE(object):
    """
        Takes a dic list and label list (returned from @return_labeled_docs_from_df()
        and return an iterator over labeled documents. Using RE tokenizer to only tokenize words
        TODO: should be refactored with the previous class
    """
    def __init__ (self, doc_list, labels_list):
        self.doc_list = doc_list
        self.labels_list = labels_list

    def __iter__(self):
        tokenizer = RegexpTokenizer(r'\w+')
        for idx, doc in zip(self.labels_list, self.doc_list):
            yield TaggedDocument(words=tokenizer.tokenize(doc), tags=[idx])