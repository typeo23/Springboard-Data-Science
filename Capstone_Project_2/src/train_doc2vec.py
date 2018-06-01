import gensim
from trainText.trainFolder import LabeledDirCsvFiles
import sys

"""
A simple testing scrip
Train for 40 epochs, the model parameters are largely the defaults
"""
#sys.stderr = open('err.txt', 'w')
assert gensim.models.doc2vec.FAST_VERSION > -1

ld = LabeledDirCsvFiles('../data/csvTrain/', 'csv')
epoches = 200

model = gensim.models.Doc2Vec(vector_size=400,  workers=6)

print('Building vocabulary.....')
model.build_vocab(ld)
model.save('../models/doc2vec/doc2vec_one_file_filter.model')

print('Training model.....')
model.train(ld, total_examples=model.corpus_count, epochs=epoches)

print('Saving model.....')
model.save('../models/doc2vec/doc2vec_one_file_filter.model')
