import gensim
from trainText.trainFolder import LabeledDir

"""
A simple testing scrip
Train for 40 epochs, the model parameters are largely the defaults
"""

ld = LabeledDir('../data/', 'gz')
epoches = 40


model = gensim.models.Doc2Vec(vector_size=400, window=15, min_count=10,
                              workers=20)

print ('Building vocabulary.....')
model.build_vocab(ld)

print ('Training model.....')
model.train(ld, total_examples=model.corpus_count, epochs=epoches)

print ('Saving model.....')
model.save('doc2vec_one_file.model')
