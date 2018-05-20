from gensim.scripts.glove2word2vec import glove2word2vec
import os
"""
Simple script to convert pretrained glove word embeddings to word2vec
for use with gensim
"""

glove_dir = '../glove.6B/'
word2vec_dir = '../glove_2_word2vec/'
glove_files = os.listdir(glove_dir)

for file in glove_files:
    print('Converting {}'.format(file))
    glove2word2vec(glove_dir+file, word2vec_dir+file)

