import pandas as pd
import os
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Model:

    def __init__(self):
        self.model = Word2Vec.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'doc2vec_freezed',
                                       'doc2vec_one_file_filter.model'))
        self.df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                           's2-corpus-002_lang_cites.csv'))
        self.result_df = pd.DataFrame([])

    def score_cites(self, df_sims, citesWeight=.1, yearWeight=1, scoreWeight=0.1):
        """
        The scoring function.
        For rach abstract the number of citations, published year and model score are scaled
        and a weighted sum defines the final score. The abstracts are then aggragates by jurnal
        and the sum of papers from each journal is multiplied by the mean score for the abstracts from
        this journal to get the final ranking

        @citesWeight : weight for the number of citations
        @yearWeight : weight for the publication year
        @scoreWeight : model score

        @returns: a pandas DataFrame with the journal ranking.
        """
        scaler = MinMaxScaler()

        scaledCites = scaler.fit_transform(np.array([df_sims.numCites]).T)
        scaledYear = scaler.fit_transform(np.array([df_sims.year]).T)
        scaledScore = scaler.fit_transform(np.array([df_sims.score]).T)
        df_sims['combinedScore'] = (
                    citesWeight * scaledCites + yearWeight * scaledYear + scoreWeight * scaledScore).flatten()

        groups = df_sims.groupby('journalName')
        df_sims2 = pd.DataFrame(groups.id.count())
        df_sims2['meanCites'] = groups.numCites.mean()
        df_sims2['meanScore'] = groups.score.mean()
        df_sims2['combinedScoreMean'] = groups.combinedScore.mean()
        df_sims2 = df_sims2[df_sims2.id > 3]
        df_sims2['finalScore'] = df_sims2.id * df_sims2.combinedScoreMean
        return df_sims2

    def find_similar(self, abstract):
        vec = self.model.infer_vector(abstract.lower().split(' '))
        sims = self.model.docvecs.most_similar(positive=[vec], topn=500)
        sims_id = [n for n, v in sims]
        sims_score = np.exp(np.array([v for n, v in sims]) * 30)
        df_sims = self.df[self.df.id.isin(sims_id)]
        df_sims['score'] = sims_score
        df_sims2 = self.score_cites(df_sims)
        self.result_df = df_sims2.sort_values(by='finalScore', ascending=False).head(20)


