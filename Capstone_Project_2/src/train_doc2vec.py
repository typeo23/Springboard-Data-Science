import gensim
import pandas as pd
from trainingFunctions import return_labeled_docs_from_df
from trainingFunctions import LabeledLineSentence

data_dir = '../data/'
epoches = 300

df = pd.read_json(data_dir+'sample-S2-records', lines=True)
df = df[df.paperAbstract != '']

labels, docs = return_labeled_docs_from_df(df)
labeledSents = LabeledLineSentence(docs, labels)

model = gensim.models.Doc2Vec(vector_size=300, window=10, min_count=5,
                              workers=11, alpha=0.025, min_alpha=0.0025)

model.build_vocab(labeledSents)


model.train(labeledSents, total_examples=model.corpus_count, epochs=epoches)


model.save('doc2vec.model')
pid1 = df.iloc[17].id
pid2 = model.docvecs.most_similar(pid1)
print(df[df.id == pid1].iloc[0].paperAbstract)
print(df[df.id == pid2[0][0]].iloc[0].paperAbstract)

