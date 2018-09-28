# NLTK imports
import nltk
from nltk.corpus import state_union
from nltk.corpus import gutenberg

import matplotlib.pyplot as plt
import numpy as np
from util import clean_text
from tqdm import tqdm
from sklearn.decomposition import PCA
import seaborn as sns

sns.set()
sns.set_context('paper')
nltk.download('state_union')
nltk.download('gutenberg')

# Set current corpus (state_union, gutenberg)
corpus = state_union

textnames = corpus.fileids()
corpusnames = {gutenberg : 'Gutenberg books', state_union : 'State union documents'}

print("cleaning texts..")
clean_texts = {name : clean_text(corpus.raw(name)) for name in tqdm(textnames)}
print("Calculating frequencies..")
freqs = {name : nltk.FreqDist(text) for name, text in tqdm(clean_texts.items())}

# Create complete vocabulary
wordlist = set()
for freq in freqs.values():
    wordlist.update(freq.keys())
vocabulary = {word : idx for (idx, word) in enumerate(wordlist)}

# Convert texts to wordcount vectors
word_vectors = {name : np.zeros(len(wordlist,)) for name in textnames}
for textname, vector in word_vectors.items():
    for word, freq in freqs[textname].items():
        vector[vocabulary[word]] += freq

# Stack text vectors into single matrix
data = np.stack(word_vectors.values())

# Apply PCA to find linear word combination pc's
_pca_ = PCA(n_components = 2)
_pca_.fit(data)
projection = _pca_.transform(data)

# Plot texts on principal component dimension
plt.scatter(projection[:, 0], projection[:, 1])
plt.title(corpusnames[corpus])
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()
