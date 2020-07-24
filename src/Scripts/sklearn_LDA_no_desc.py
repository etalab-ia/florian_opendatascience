#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition
import pyLDAvis.sklearn
import warnings
from collections import Counter

from tqdm import tqdm

from resources.locs import LOCS, SPECIAL_WORDS

warnings.filterwarnings('ignore')

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import spacy

nlp = spacy.load("fr_core_news_lg")
from spacy.lang.fr.stop_words import STOP_WORDS
from spacy.lang.fr import French
import string

punctuations = string.punctuation
stopwords = list(STOP_WORDS)

data = pd.read_pickle("../../data/nodesc_clean.pkl")
data_np = data.to_numpy()

data_np_noloc = np.load('../../data/data_np_noloc.npy', allow_pickle=True)

# #### LDA

stopwords_full = SPECIAL_WORDS + list(LOCS) + stopwords
def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [word.lower_ for word in tokens]
    tokens = [word for word in tokens if word not in stopwords_full]
    tokens = " ".join([i for i in tokens])
    return tokens
#
#
parser = French()

def basic_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [word.lower_ for word in tokens]
    tokens = [word for word in tokens if word not in stopwords]
    tokens = " ".join([i for i in tokens])
    return tokens
#
#
data_np_nostop = []
for i in tqdm(range(len(data_np))):
    data_np_nostop.append(basic_tokenizer(data_np_noloc[i]))


producteurs_np = pd.read_pickle("../../data/prod_set.pkl").to_numpy()
producteurs_np = producteurs_np.transpose()[0]

def remove_locs(doc):
    all_docs = [None] * len(doc)
    for i in tqdm(range(len(doc))):
        this_doc = []
        for token in nlp(doc[i]):
            if token.ent_type_ == 'LOC':
                continue
            else:
                this_doc.append(token.text)
        all_docs[i] = ' '.join(this_doc)
    return (all_docs)


producteurs_np1 = remove_locs(producteurs_np)

producteurs_np_clean = []
for i in tqdm(range(len(producteurs_np1))):
    producteurs_np_clean.append(basic_tokenizer(producteurs_np1[i]))

producteurs_np_clean

vectorizer_prod = CountVectorizer(min_df=3, max_df=5000, lowercase=True, strip_accents='unicode', ngram_range=(1, 2))
data_vectorized_prod = vectorizer_prod.fit_transform(producteurs_np_clean)

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=40, max_iter=50, evaluate_every=5, verbose=True)
data_lda = lda.fit_transform(data_vectorized_prod)

# Keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
               for i in topic.argsort()[:-top_n - 1:-1]])



print("LDA Model:")
selected_topics(lda, vectorizer_prod)

# In[160]:


pyLDAvis.enable_notebook()
dash = pyLDAvis.sklearn.prepare(lda, data_vectorized_prod, vectorizer_prod, mds='mmds')
dash

pyLDAvis.save_html(dash, 'lda_p_40_v2_mmds.html')

vectorizer = CountVectorizer(min_df=10, max_df=5000, lowercase=True, strip_accents='unicode', ngram_range=(1, 2))
data_vectorized = vectorizer.fit_transform(data_np_nostop)

NUM_TOPICS = 30


lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=50, evaluate_every=5, verbose=True)
data_lda = lda.fit_transform(data_vectorized)

np.save('../../data/data_ldav5_nondesc_40.npy', data_lda, allow_pickle=True)


data_lda = np.load('../../data/data_ldav4_nondesc_40.npy', allow_pickle=True)

# Keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
               for i in topic.argsort()[:-top_n - 1:-1]])

print("LDA Model:")
selected_topics(lda, vectorizer)

pyLDAvis.enable_notebook()
dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='mmds')
dash

pyLDAvis.save_html(dash, 'lda_30_mmds.html')

# Transforming an individual sentence
text = spacy_tokenizer("reseau transport bus horaires adresse voirie arrêts")
x = lda.transform(vectorizer.transform([text]))[0]
print(x)
