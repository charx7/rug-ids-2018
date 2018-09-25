# NLTK imports
import nltk
from nltk.corpus import webtext
from nltk.corpus import state_union
from nltk.probability import DictionaryProbDist

import matplotlib.pyplot as plt
import numpy as np
from util import clean_text

# Import the vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


nltk.download('state_union')
nltk.download('stopwords')
nltk.download('punkt')

textnames = state_union.fileids()

print("cleaning texts..")
clean_texts = {name : clean_text(state_union.raw(name)) for name in textnames}
freqs = {name : nltk.FreqDist(text) for name, text in clean_texts.items()}
print("done")

# Create complete vocabulary
vocabulary = []
for freq in freqs.values():
    vocabulary += list(freq.keys())
vocab_range = list(range(len(vocabulary)))
vocabulary = dict(zip(vocabulary, vocab_range))
print(vocabulary)

# Convert texts to wordcount vectors
word_vectors = {name : np.zeros(len(vocabulary,)) for name in textnames}
for textname, vector in word_vectors.items():
    for word, freq in freqs[textname].items():
        # The word count at a fixed index is increased by one
        vector[vocabulary[word]] += freq

data = np.stack(word_vectors.values())
print(mat.shape)

_pca_ = PCA(n_components = 2)
_pca_.fit(data)
projection = _pca_.transform(data)
print(projection.shape)

plt.scatter(projection[:, 0], projection[:, 1])
