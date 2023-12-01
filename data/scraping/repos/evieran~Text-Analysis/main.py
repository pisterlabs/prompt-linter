# Bring in text file with our novel
textfile = open('great_expectations.txt', 'r', encoding = "utf8")
great_expect = textfile.read()

# Or import the text file
path_to_file = '/Users/yourusername/Downloads/great_expectations.txt'

with open(path_to_file, 'r', encoding='utf8') as textfile:
    great_expect = textfile.read()

print(great_expect)

# Import libraries
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud

import pandas as pd
from PIL import Image
import numpy as np
import random
import re
import matplotlib.pyplot as plt
%matplotlib inline

# Lowercase words for word cloud
word_cloud_text = great_expect.lower()
# Remove numbers and alphanumeric words we don't need for word cloud
word_cloud_text = re.sub("[^a-zA-Z0-9]", " ", word_cloud_text)

# Tokenize the data to split it into words
tokens = word_tokenize(word_cloud_text)

# Remove stopwords
tokens = (word for word in tokens if word not in stopwords.words("english"))

# Remove short words less than 3 letters in lenght
tokens = (word for word in tokens if len(word) >=3)

# Data cleaning to split data into sentences
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"

text = " " + great_expect + "  "
text = text.replace("\n"," ")
text = re.sub(prefixes,"\\1<prd>",text)
text = re.sub(websites,"<prd>\\1",text)
text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
if "..." in text: text = text.replace("...","<prd><prd><prd>")
if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
if "”" in text: text = text.replace(".”","”.")
if "\"" in text: text = text.replace(".\"","\".")
if "!" in text: text = text.replace("!\"","\"!")
if "?" in text: text = text.replace("?\"","\"?")
text = text.replace(".",".<stop>")
text = text.replace("?","?<stop>")
text = text.replace("!","!<stop>")
text = text.replace("<prd>",".")
sentences = text.split("<stop>")
sentences = [s.strip() for s in sentences]
sentences = pd.DataFrame(sentences)
sentences.columns = ['sentence']

# Print out sentences variable
print(len(sentences))
print(sentences.head(10))

# Remove the first few rows of text that are irrelevant for analysis
sentences.drop(sentences.index[:59], inplace=True)
sentences = sentences.reset_index(drop=True)
sentences.head(10)

# Create word cloud with our text data
stopwords_wc = set(stopwords.words("english"))

wordcloud = WordCloud(max_words=100, stopwords = stopwords_wc, random_state = 1).generate(word_cloud_text)
plt.figure(figsize=(12,16))
plt.imshow(wordcloud)
plt.axis("off")
plt.show

# How to improve a word cloud
mask = np.array(Image.open("man_in_top_hat.jpeg"))
def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60,100)

# Create advanced Word Cloud with our text data
wordcloud = WordCloud(background_color = "purple", max_words=100, mask=mask, color_func = grey_color_func, stopwords = stopwords_wc, random_state = 1).generate(word_cloud_text)
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.show

# Visualization of top 50 most common wods in text
plt.figure(figsize=(12,6))
fdist.plot(50)
plt.show()

# How to perform Vader sentiment analysis
analyzer = SentimentIntensityAnalyzer()
sentences['compound'] = [analyzer.polarity_scores(x)['compound'] for x in sentences['sentence']]
sentences['neg'] = [analyzer.polarity_scores(x)['neg'] for x in sentences['sentence']]
sentences['neu'] = [analyzer.polarity_scores(x)['neu'] for x in sentences['sentence']]
sentences['pos'] = [analyzer.polarity_scores(x)['pos'] for x in sentences['sentence']]
sentences.head(10)

#Get number of positive, negative, and neutral sentences
positive_sentence = sentences.loc[sentences['compound'] > 0]
negative_sentence = sentences.loc[sentences['compound'] < 0]
neutral_sentence = sentences.loc[sentences['compound'] == 0]

print(sentences.shape)
print(positive_sentence)
print(negative_sentence)
print(neutral_sentence)

# Convert sentence data to list
data = sentences['sentence'].values.tolist()
type(data)

# Text cleaning and tokenization using function
def text_processing(texts):
    #Remove numbers and alphanumerical words we don't need
    texts =  [re.sub("[^a-zA-Z]+", " ", str(text)) for text in texts]
    #Tokenize & lowercase each word
    texts = [[word for word in text.lower().split()] for text in texts]
    #Stem each word
    lmtzr = WordNetLemmatizer()
    texts = [[lmtzr.lemmatize(word) for word in text] for text in texts]
    #Remove stopwords
    stoplist = stopwords.words('english')
    texts = [[word for word in text if word not in stoplist] for text in texts]
    #Remove short words less than 3 letters in length
    texts = [[word for word in tokens if len(word) >= 3] for tokens in texts]
    return texts

# Apply function to process data and convert to dictionary
data = text_processing(data)
dictionary = Dictionary(data)
len(dictionary)

# Create corpus for LDA analysis
corpus = [dictionary.doc2bow(text) for text in data]
len(corpus)

# How to perform topic modeling
# Find optimal k value for the number of topics for our LDA analysis
np.random.seed(1)
k_range = range(6,20,2)
scores = []
for k in k_range:
    LdaModel = ldamodel.LdaModel(corpus = corpus, id2word = dictionary, num_topics = k, passes = 20)
    cm = CoherenceModel(model= LdaModel, corpus = corpus, dictionary = dictionary, coherence = 'u_mass')
    print(cm.get_coherence())
    scores.append(cm.get_coherence())

plt.figure()
plt.plot(k_range, scores)

#Build LDA topic model
model = ldamodel.LdaModel(corpus, id2word = dictionary, num_topics = 6, passes = 20)

#Print topic distribution
model.show_topics()
