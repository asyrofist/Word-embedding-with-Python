import streamlit as st
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

def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
#         lexicon.update([word for word in doc.split()])
        lexicon.update([word for word in doc])
    return lexicon

# preprocess
st.header("Word2Vec")
sentences = list(gutenberg.sents('shakespeare-hamlet.txt'))   # import the corpus and convert into a list
for i in range(len(sentences)):
    sentences[i] = [word.lower() for word in sentences[i] if re.match('^[a-zA-Z]+', word)]
st.dataframe(sentences)

# Model
st.sidebar.subheader("Model Parameter")
mode_value = st.sidebar.selectbox('What mode?',[0, 1])
size_value = st.sidebar.slider('What mode?', 0, 1000, 100)
window_value = st.sidebar.slider('What mode?', 0, 10, 3)
iteration_value = st.sidebar.slider('What mode?', 0, 100, 10)

model = Word2Vec(sentences = sentences, size = size_value, sg = mode_value, window = window_value, min_count = 1, iter = iteration_value, workers = Pool()._processes)
model.init_sims(replace = True)
model.save('word2vec_model')
model = Word2Vec.load('word2vec_model')

# evaluation
vocabulary = build_lexicon(sentences)
options1 = st.multiselect('What word do you choose?',vocabulary)
st.write(options1)
options2 = st.multiselect('What word do you choose?',vocabulary)
st.write(options2)
# hasil_cosine = cosine_similarity(model[options1], model[options2])
# st.write(hasil_cosine)
