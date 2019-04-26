import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
import pickle
from multiprocessing import Pool


def get_mean_abstract_vector(abstract):
    tokens = nltk.word_tokenize(abstract.lower())
    tokens = [token for token in tokens if (token not in stopwords.words('english') and token in model.wv.vocab)]
    return np.mean(model[tokens], axis=0)


file_dir ='../data/csvTrain/'
filename ='s2-corpus-002_lang.csv'
model =gensim.models.Word2Vec.load('../models/word2vec/word2vec_one_file_filter.model')
df = pd.read_csv(file_dir+filename)
with Pool(18) as p:
    mean_scores = p.map(get_mean_abstract_vector, df.paperAbstract.iloc[1:-10])

with open('../models/word2vec/word2vecMeans.pk', 'wb') as fname:
    pickle.dump(np.array(mean_scores), fname)

print(len(mean_scores))
print(np.array(mean_scores).shape)