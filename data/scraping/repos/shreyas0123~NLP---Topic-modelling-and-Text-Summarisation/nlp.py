##################### problem1 #######################
#1)	Perform NLP – Topic Modelling and Text summarization by following all the steps as mentioned below: -
#2)	Data Cleaning using regular expressions, Count Vectorizer, POS Tagging, NER, Topic Modelling (LDA, LSA) and Text summarization.
#Hint: - Use Data.csv file given in hands on material.

import pandas as pd

twitter = pd.read_csv("C://Users//DELL//Downloads//Data.csv",usecols = ['text'])
twitter.head(10)

#data cleansing using regular expression, Count Vectorizer, POS Tagging, NER
import re

HANDLE = '@\w+'
LINK = 'https?://t\.co/\w+'
SPECIAL_CHARS = '&lt;|&lt;|&amp;|#'

def clean(text):
    text = re.sub(HANDLE, ' ', text)
    text = re.sub(LINK, ' ', text)
    text = re.sub(SPECIAL_CHARS, ' ', text)
    return text

twitter['text'] = twitter.text.apply(clean)
twitter.head(10)

tweet = []
for i in twitter['text']:
    tweet.append(i)


# Word Tokenization 
import nltk
nltk.download("punkt")
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize

token = word_tokenize("tweet")
print(twitter)

nltk.pos_tag(twitter) # Parts of Speech Tagging

nltk.download('stopwords')  # Stop Words from nltk library
from nltk.corpus import stopwords
stop_words = stopwords.words('English') # 179 pre defined stop words
print(stop_words)

sentence_no_stops = ' '.join([word for word in tweet if word not in stop_words]) 
print(tweet)

# Stemming
stemmer = nltk.stem.PorterStemmer()
stemmer.stem("tweet")

# Lemmatization
# Lemmatization looks into dictionary words
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('tweet')

# LDA
from gensim.parsing.preprocessing import preprocess_string

twitter = twitter.text.apply(preprocess_string).tolist()

from gensim import corpora
from gensim.models.ldamodel import LdaModel

dictionary = corpora.Dictionary(twitter)
corpus = [dictionary.doc2bow(text) for text in twitter]

NUM_TOPICS = 5
ldamodel = LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=10)

ldamodel.print_topics(num_words=5)

from gensim.models.coherencemodel import CoherenceModel

def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()

def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        ldamodel = LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=2)
        coherence = calculate_coherence_score(twitter, dictionary, ldamodel)
        yield coherence
        
        
min_topics, max_topics = 10,16
coherence_scores = list(get_coherence_values(min_topics, max_topics))

import matplotlib.pyplot as plt       

x = [int(i) for i in range(min_topics, max_topics)]

ax = plt.figure(figsize=(10,8))
plt.xticks(x)
plt.plot(x, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence Value')
plt.title('Coherence Scores', fontsize=10);

################ LSA #################################
# Topic Modelling
# Latent Semantic Analysis / Latent Semantic Indexing

from gensim import corpora # Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora.
from gensim.models import LsiModel
from gensim.parsing.preprocessing import preprocess_string

import re

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    x = re.sub(pattern, '', x)
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

def clean(x):
    x = clean_text(x)
    x = clean_numbers(x)
    return x

from pathlib import Path
from bs4 import BeautifulSoup

def load_articles(data_dir):
    reuters = Path(data_dir)
    for path in reuters.glob('*.sgm'):   # Standard Generalized Markup Language
        with path.open() as sgm_file:
            contents = sgm_file.read()
            soup = BeautifulSoup(contents)
            for article in soup.find_all('body'):
                yield article.text

def load_documents(document_dir):
    print(f'Loading from {document_dir}')
    documents = list(load_articles(document_dir))
    print(f'Loaded {len(documents)} documents')
    return documents

def prepare_documents(documents):
    print('Preparing documents')
    documents = [clean(document) for document in documents]
    documents = [preprocess_string(doc) for doc in documents]
    return documents

def create_lsa_model(documents, dictionary, number_of_topics):
    print(f'Creating LSA Model with {number_of_topics} topics')
    document_terms = [dictionary.doc2bow(doc) for doc in documents]
    return LsiModel(document_terms, num_topics=number_of_topics, id2word = dictionary)

def run_lsa_process(documents, number_of_topics=10):
    documents = prepare_documents(documents)
    dictionary = corpora.Dictionary(documents)
    lsa_model = create_lsa_model(documents, dictionary, number_of_topics)
    return documents, dictionary, lsa_model

# data directory 
twitter = pd.read_csv("C://Users//DELL//Downloads//Data.csv",usecols = ['text'])
documents, dictionary, model = run_lsa_process(twitter['text'], number_of_topics=5)

model.print_topics()
model

# Coherence Model
from gensim.models.coherencemodel import CoherenceModel

def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()


def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        documents, dictionary, model = run_lsa_process(twitter['text'], number_of_topics=num_topics)
        coherence = calculate_coherence_score(documents, dictionary, model)
        yield coherence

min_topics, max_topics = 5, 11

coherence_scores = list(get_coherence_values(min_topics, max_topics))
documents

## Plot
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')

x = [int(i) for i in range(min_topics, max_topics)]

plt.figure(figsize=(10,8))
plt.plot(x, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence Value')
plt.title('Coherence Scores by number of Topics')


####################### Text Summarization #############################
import nltk
nltk.download('stopwords')

from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest

STOPWORDS = set(stopwords.words('english') + list(punctuation))
MIN_WORD_PROP, MAX_WORD_PROP = 0.1, 0.9

#####
def compute_word_frequencies(word_sentences):
    words = [word for sentence in word_sentences 
                     for word in sentence 
                         if word not in STOPWORDS]
    counter = Counter(words)
    limit = float(max(counter.values()))
    word_frequencies = {word: freq/limit 
                                for word,freq in counter.items()}
    # Drop words if too common or too uncommon
    word_frequencies = {word: freq 
                            for word,freq in word_frequencies.items() 
                                if freq > MIN_WORD_PROP 
                                and freq < MAX_WORD_PROP}
    return word_frequencies
####

####
def sentence_score(word_sentence, word_frequencies):
    return sum([ word_frequencies.get(word,0) 
                    for word in word_sentence])
###
    
###
def summarize(text:str, num_sentences=3):
    """
    Summarize the text, by return the most relevant sentences
     :text the text to summarize
     :num_sentences the number of sentences to return
    """
    text = text.lower() # Make the text lowercase
    
    sentences = sent_tokenize(text) # Break text into sentences 
    
    # Break sentences into words
    word_sentences = [word_tokenize(sentence) for sentence in sentences]
    
    # Compute the word frequencies
    word_frequencies = compute_word_frequencies(word_sentences)
    
    # Calculate the scores for each of the sentences
    scores = [sentence_score(word_sentence, word_frequencies) for word_sentence in word_sentences]
    sentence_scores = list(zip(sentences, scores))
    
    # Rank the sentences
    top_sentence_scores = nlargest(num_sentences, sentence_scores, key=lambda t: t[1])
    
    # Return the top sentences
    return [t[0] for t in top_sentence_scores]
###
twitter = pd.read_csv("C://Users//DELL//Downloads//Data.csv",usecols = ['text'])   

tweet = ' '.join([word for word in twitter['text']])

len(sent_tokenize(tweet))

summarize(tweet)

summarize(tweet, num_sentences=1)

############################## problem2 ############################
#2)Perform topic modelling and text summarization on the given text data hint use NLP-TM text file.

import pandas as pd

text_data=  open("C:/Users/DELL/Downloads/NLP-TM.txt",encoding = "utf8")

from nltk.corpus import stopwords

stop_words = stopwords.words('English') # 179 pre defined stop words
print(stop_words)

txt = ' '.join([word for word in text_data if word not in stop_words])
txt

# Latent Dirichlet Allocation
from nltk.tokenize import sent_tokenize
sent_token_article = sent_tokenize(txt)
from pandas import DataFrame
sent_token_article = DataFrame(sent_token_article, columns=['text'])

import re

HANDLE = '@\w+'
LINK = 'https?://t\.co/\w+'
SPECIAL_CHARS = '&lt;|&lt;|&amp;|#'

def clean(sent_token_article):
    text = re.sub(HANDLE, ' ', sent_token_article)
    text = re.sub(LINK, ' ', sent_token_article)
    text = re.sub(SPECIAL_CHARS, ' ', sent_token_article)
    return sent_token_article

sent_token_article['text'] = sent_token_article.text.apply(clean)
sent_token_article.head(10)

# LDA
from gensim.parsing.preprocessing import preprocess_string

sent_token_article = sent_token_article.text.apply(preprocess_string).tolist()

from gensim import corpora
from gensim.models.ldamodel import LdaModel

dictionary = corpora.Dictionary(sent_token_article)
corpus = [dictionary.doc2bow(text) for text in sent_token_article]

NUM_TOPICS = 5
ldamodel = LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=10)

ldamodel.print_topics(num_words=5)

from gensim.models.coherencemodel import CoherenceModel

def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()

def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        ldamodel = LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=2)
        coherence = calculate_coherence_score(sent_token_article, dictionary, ldamodel)
        yield coherence


min_topics, max_topics = 10,16
coherence_scores = list(get_coherence_values(min_topics, max_topics))

import matplotlib.pyplot as plt
# import matplotlib.style as style

# get_ipython().run_line_magic('matplotlib', 'auto') # will give us the plots inline only

x = [int(i) for i in range(min_topics, max_topics)]

ax = plt.figure(figsize=(10,8))
plt.xticks(x)
plt.plot(x, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence Value')
plt.title('Coherence Scores', fontsize=10);

####################### LSA #####################################
# Topic Modelling
# Latent Semantic Analysis / Latent Semantic Indexing

# pip install gensim

from gensim import corpora # Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora.
from gensim.models import LsiModel
from gensim.parsing.preprocessing import preprocess_string

import re

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    x = re.sub(pattern, '', x)
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

def clean(x):
    x = clean_text(x)
    x = clean_numbers(x)
    return x

from pathlib import Path
from bs4 import BeautifulSoup

def load_articles(data_dir):
    reuters = Path(data_dir)
    for path in reuters.glob('*.sgm'):   # Standard Generalized Markup Language
        with path.open() as sgm_file:
            contents = sgm_file.read()
            soup = BeautifulSoup(contents)
            for article in soup.find_all('body'):
                yield article.text

def load_documents(document_dir):
    print(f'Loading from {document_dir}')
    documents = list(load_articles(document_dir))
    print(f'Loaded {len(documents)} documents')
    return documents

def prepare_documents(documents):
    print('Preparing documents')
    documents = [clean(document) for document in documents]
    documents = [preprocess_string(doc) for doc in documents]
    return documents

def create_lsa_model(documents, dictionary, number_of_topics):
    print(f'Creating LSA Model with {number_of_topics} topics')
    document_terms = [dictionary.doc2bow(doc) for doc in documents]
    return LsiModel(document_terms, num_topics=number_of_topics, id2word = dictionary)

def run_lsa_process(documents, number_of_topics=10):
    documents = prepare_documents(documents)
    dictionary = corpora.Dictionary(documents)
    lsa_model = create_lsa_model(documents, dictionary, number_of_topics)
    return documents, dictionary, lsa_model

# data directory 
text_data=  open("C:/Users/DELL/Downloads/NLP-TM.txt",encoding = "utf8")

from nltk.corpus import stopwords

stop_words = stopwords.words('English') # 179 pre defined stop words
print(stop_words)

txt = ' '.join([word for word in text_data if word not in stop_words])
txt

from nltk.tokenize import sent_tokenize
sent_token_article = sent_tokenize(txt)
from pandas import DataFrame
sent_token_article = DataFrame(sent_token_article, columns=['text'])

#loading LSA model
documents, dictionary, model = run_lsa_process(sent_token_article['text'], number_of_topics=5)

model.print_topics()
model

# Coherence Model
from gensim.models.coherencemodel import CoherenceModel

def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()


def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        documents, dictionary, model = run_lsa_process(sent_token_article, number_of_topics=num_topics)
        coherence = calculate_coherence_score(documents, dictionary, model)
        yield coherence

min_topics, max_topics = 5, 11

coherence_scores = list(get_coherence_values(min_topics, max_topics))
documents

## Plot
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')

x = [int(i) for i in range(min_topics, max_topics)]

plt.figure(figsize=(10,8))
plt.plot(x, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence Value')
plt.title('Coherence Scores by number of Topics')

####################### text summarization #############################
# Text Summarization

import nltk
nltk.download('stopwords')

from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest

STOPWORDS = set(stopwords.words('english') + list(punctuation))
MIN_WORD_PROP, MAX_WORD_PROP = 0.1, 0.9

#####
def compute_word_frequencies(word_sentences):
    words = [word for sentence in word_sentences 
                     for word in sentence 
                         if word not in STOPWORDS]
    counter = Counter(words)
    limit = float(max(counter.values()))
    word_frequencies = {word: freq/limit 
                                for word,freq in counter.items()}
    # Drop words if too common or too uncommon
    word_frequencies = {word: freq 
                            for word,freq in word_frequencies.items() 
                                if freq > MIN_WORD_PROP 
                                and freq < MAX_WORD_PROP}
    return word_frequencies
####

####
def sentence_score(word_sentence, word_frequencies):
    return sum([ word_frequencies.get(word,0) 
                    for word in word_sentence])
###
    
###
def summarize(text:str, num_sentences=3):
    """
    Summarize the text, by return the most relevant sentences
     :text the text to summarize
     :num_sentences the number of sentences to return
    """
    text = text.lower() # Make the text lowercase
    
    sentences = sent_tokenize(text) # Break text into sentences 
    
    # Break sentences into words
    word_sentences = [word_tokenize(sentence) for sentence in sentences]
    
    # Compute the word frequencies
    word_frequencies = compute_word_frequencies(word_sentences)
    
    # Calculate the scores for each of the sentences
    scores = [sentence_score(word_sentence, word_frequencies) for word_sentence in word_sentences]
    sentence_scores = list(zip(sentences, scores))
    
    # Rank the sentences
    top_sentence_scores = nlargest(num_sentences, sentence_scores, key=lambda t: t[1])
    
    # Return the top sentences
    return [t[0] for t in top_sentence_scores]
###
    

import pandas as pd

text_data=  open("C:/Users/DELL/Downloads/NLP-TM.txt",encoding = "utf8")
my_article = ' '.join([word for word in text_data])

len(sent_tokenize(my_article))

summarize(my_article)

summarize(my_article, num_sentences=1)

















