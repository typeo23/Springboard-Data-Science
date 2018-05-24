import os
import pandas as pd
from trainText.trainingFunctions import return_labeled_docs_from_df
from nltk.tokenize import RegexpTokenizer
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
            reader = pd.read_json(os.path.join(self.files_dir, file), lines=True, chunksize=300)
            for df in reader:
                labels, docs = return_labeled_docs_from_df(df)
                tokenizer = RegexpTokenizer(r'\w+')
                for idx, doc in zip(labels, docs):
                    yield TaggedDocument(words=tokenizer.tokenize(doc), tags=[idx])