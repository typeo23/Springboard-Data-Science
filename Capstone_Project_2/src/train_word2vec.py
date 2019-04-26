# import modules & set up logging
import gensim, logging
from trainText.trainFolder import DirSentenceTokenizerCsvFiles

"""
A simple testing scrip
Train for 40 epochs, the model parameters are largely the defaults
"""
assert gensim.models.doc2vec.FAST_VERSION > -1
epochs = 50

sentences = DirSentenceTokenizerCsvFiles('../data/csvTrain/', 'csv')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, workers=4, iter=epochs, min_count=4, size=200)
model.save('../models/word2vec/doc2vec_one_file_filter.model')