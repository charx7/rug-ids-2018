#Importing NLTK and download : tokenizer, tagger, stopwords, corpus
import nltk
import pandas as pd
from nltk.corpus import stopwords, state_union
from nltk.tokenize import word_tokenize
from nltk.collocations import TrigramCollocationFinder
from nltk import pos_tag
nltk.download('state_union')
nltk.download('stopwords')
nltk.download('tagsets')


#make corpusList ready
corpusList = []
for i in range(len(state_union.fileids())):
    corpusList.append(state_union.raw(state_union.fileids()[i]))

#concatanete all raw texts within corpusList
allTexts = " ".join(corpusList)

#get english stop words
stop_words = set(stopwords.words('english'))
#get a sample raw txt from state_union corpus
#stateUnionRawExp = state_union.raw(rawText)
tokens = word_tokenize(allTexts)
#tag tokens
tagged = pos_tag(tokens)
#convert tagged tuple into dataframe for the ease of manipulation
tagged = pd.DataFrame(list(tagged), columns=["word","type"])
#turn words into lowercase except NNP and NNPS
for i in range(len(tagged)):
    if tagged["type"][i] in ('NNP','NNPS'):
        pass
    else:
        tagged["word"][i] = tagged["word"][i].lower()
#filter out stopwords and puncuations
stopWordFiltered = [w for w in tagged["word"].values if not w in stop_words]
puncFiltered = [w for w in stopWordFiltered if len(w)>2]
#tag tokens again
filteredTagged = pos_tag(puncFiltered)
#find trigrams
finder = TrigramCollocationFinder.from_words(filteredTagged,3)
#obtain frequency distribution for all trigrams
freqDistTrigram = finder.ngram_fd.items()
freqDistTable = pd.DataFrame(list(freqDistTrigram), columns=['collocations','freq']).sort_values(by='freq', ascending=False)
#filter for NNP and NNPS last time

#return top 10 results as dataframe
