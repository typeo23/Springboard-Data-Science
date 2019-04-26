from sklearn.feature_extraction.text import TfidfVectorizer
from trainText.trainFolder import DirStreamAbstractCSV
import pickle

"""
A simple testing scrip
Train for 40 epochs, the model parameters are largely the defaults
"""

sentences = DirStreamAbstractCSV('../data/csvTrain/', 'csv')


# train word2vec on the two sentences
model = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_df=0.3, min_df=1e-4)
model.fit(sentences)
with open('../models/tfidf/tfidf.pk', 'wb') as fname:
    pickle.dump(model, fname)

print('Transforming Abstracts')
vectorized_abstracts = model.transform(sentences)
with open('../models/tfidf/tfidfAbstracts.pk', 'wb') as fname:
    pickle.dump(vectorized_abstracts, fname)


#model.save('../models/tfidf/doc2vec_one_file_filter.model')