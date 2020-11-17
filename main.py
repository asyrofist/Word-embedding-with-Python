import streamlit
import re
import numpy as np
import nltk
from nltk.corpus import gutenberg
from gensim.models import Word2Vec
from multiprocessing import Pool
from scipy import spatial

nltk.download('gutenberg')
nltk.download('punkt')

# define a function that computes cosine similarity between two words
def cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)

# preprocess
st.header("Word2Vec")
sentences = list(gutenberg.sents('shakespeare-hamlet.txt'))   # import the corpus and convert into a list
for i in range(len(sentences)):
    sentences[i] = [word.lower() for word in sentences[i] if re.match('^[a-zA-Z]+', word)]
st.write(sentences)

# Model
model = Word2Vec(sentences = sentences, size = 100, sg = 1, window = 3, min_count = 1, iter = 10, workers = Pool()._processes)
model.init_sims(replace = True)
model.save('word2vec_model')
model = Word2Vec.load('word2vec_model')

# evaluation
model.most_similar('hamlet')
v1 = model['king']
v2 = model['queen']
hasil_cosine = cosine_similarity(v1, v2)
st.write(hasil_cosine)
