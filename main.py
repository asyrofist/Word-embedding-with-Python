import streamlit as st
import re
import numpy as np
import nltk
from nltk.corpus import gutenberg
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
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
        lexicon.update([word for word in doc])
    return lexicon

st.header("Word2Vec")
dataset_vec = st.selectbox('What dataset do you use?',['guttenberg', 'google'])

if dataset_vec == 'guttenberg':
    # preprocess
    col1, col2 = st.beta_columns([3,1])
    col1.subheader("Dataset")
    sentences = list(gutenberg.sents('shakespeare-hamlet.txt'))   # import the corpus and convert into a list
    for i in range(len(sentences)):
        sentences[i] = [word.lower() for word in sentences[i] if re.match('^[a-zA-Z]+', word)]
    col1.dataframe(sentences)

    # vocabulary
    col2.subheader("Vocabulary")
    vocabulary = build_lexicon(sentences)
    kata = [word for word in vocabulary]
    col2.dataframe(kata)

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

    kata_value = st.selectbox('What mode?',kata)
    hasil = model.most_similar(kata_value)
    st.dataframe(hasil)

if  dataset == 'google':
    # preprocess
    col1, col2 = st.beta_columns([3,1])
    col1.subheader("Dataset")
    sentences = list(gutenberg.sents('shakespeare-hamlet.txt'))   # import the corpus and convert into a list
    for i in range(len(sentences)):
        sentences[i] = [word.lower() for word in sentences[i] if re.match('^[a-zA-Z]+', word)]
    col1.dataframe(sentences)

    # vocabulary
    col2.subheader("Vocabulary")
    vocabulary = build_lexicon(sentences)
    kata = [word for word in vocabulary]
    col2.dataframe(kata)

    # Model
    st.sidebar.subheader("Model Parameter")
    mode_value = st.sidebar.selectbox('What mode?',[0, 1])
    size_value = st.sidebar.slider('What mode?', 0, 1000, 100)
    window_value = st.sidebar.slider('What mode?', 0, 10, 3)
    iteration_value = st.sidebar.slider('What mode?', 0, 100, 10)
    
    filename = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(filename, binary=True)

    kata_value = st.selectbox('What mode?',kata)
    hasil = model.most_similar(kata_value)
    st.dataframe(hasil)
