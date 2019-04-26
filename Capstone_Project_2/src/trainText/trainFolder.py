import os
import pandas as pd
from trainText.trainingFunctions import return_labeled_docs_from_df
import nltk
from gensim.models.doc2vec import TaggedDocument

class LabeledDir(object):
    """
    A class which returns an iterable over  all files in a folder
    The files contains line jason files downloaded from semantic scholar
    It yields an iterator over tagged abstracts (a tagged document containing a tokenized abstract
    and a document ID
    """
    def __init__(self, files_dir='', file_ext=''):
        """
        :param files_dir: Directory containing the files
        :param file_ext: The files extensions. Will read all the files with this extension in  files_dir
        //TODO: add chunksize control
        """
        self.files_dir = files_dir
        self.file_ext = file_ext

    def __iter__(self):
        """
        streaming iterator over all the files in the folder, reads them in chunks so evrything fits into memory
        :return: yields and iterator over all the abstracts in the json files in the folder.
        //TODO: Test different tokenizer types (Maybe the RE is a bottle neck)
        """
        files = [f for f in os.listdir(self.files_dir) if f.split('.')[1] == self.file_ext]
        for file in files:
            print(file)
            reader = pd.read_json(os.path.join(self.files_dir, file), lines=True, chunksize=500)
            for df in reader:
                labels, docs = return_labeled_docs_from_df(df)
                #tokenizer = RegexpTokenizer(r'\w+')
                for idx, doc in zip(labels, docs):

                    yield TaggedDocument(words=doc.lower().split(), tags=[idx])


class LabeledDirCsvFiles(object):
    """
    streaming iterator over all CSV files in the folder, assuming each file fits in memory
    :return: yields and iterator over all the abstracts in the json files in the folder.
    """
    def __init__(self, files_dir='', file_ext='', only_english = True):
        self.files_dir = files_dir
        self.file_ext = file_ext
        self.only_english = only_english
        self.epoch=0

    def __iter__(self):
        files = [f for f in os.listdir(self.files_dir) if f.split('.')[1] == self.file_ext]
        for file in files:
            print(str(self.epoch) + ' ' + file)
            self.epoch = self.epoch + 1
            df = pd.read_csv(os.path.join(self.files_dir, file))
            if self.only_english:
                df = df[df.lang == 'English']
            labels, docs = return_labeled_docs_from_df(df)
            for idx, doc in zip(labels, docs):

                yield TaggedDocument(words=doc.lower().split(), tags=[idx])


class DirSentenceTokenizerCsvFiles(object):
    """
    streaming iterator over all CSV files in the folder, for Word2Vec training.
    :return: yields and iterator whicih returns tokenized sentances for each abstract.
    """
    def __init__(self, files_dir='', file_ext='', only_english = True):
        self.files_dir = files_dir
        self.file_ext = file_ext
        self.only_english = only_english
        self.epoch=0

    def __iter__(self):
        files = [f for f in os.listdir(self.files_dir) if f.split('.')[1] == self.file_ext]
        for file in files:
            print(str(self.epoch) + ' ' + file)
            self.epoch = self.epoch + 1
            df = pd.read_csv(os.path.join(self.files_dir, file))
            if self.only_english:
                df = df[df.lang == 'English']
            for abstract in df.paperAbstract.values:
                for sentence in abstract.split('.'):
                    yield nltk.word_tokenize(sentence.lower())


class DirStreamAbstractCSV(object):
    def __init__(self, files_dir='', file_ext='', only_english = True):
        self.files_dir = files_dir
        self.file_ext = file_ext
        self.only_english = only_english
        self.epoch=0

    def __iter__(self):
        files = [f for f in os.listdir(self.files_dir) if f.split('.')[1] == self.file_ext]
        for file in files:
            print(str(self.epoch) + ' ' + file)
            self.epoch = self.epoch + 1
            df = pd.read_csv(os.path.join(self.files_dir, file))
            if self.only_english:
                df = df[df.lang == 'English']
            for abstract in df.paperAbstract.values:
                yield abstract
