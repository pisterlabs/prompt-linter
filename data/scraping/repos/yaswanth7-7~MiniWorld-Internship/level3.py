import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import wordnet

# Collect input from the user
user_input = input("Enter your preferred industry keywords or general topic preferences: ")

# Retrieve trending topics from Twitter
url = "https://twitter.com/i/trends"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
trends = soup.select("[data-testid='trend'] .css-901oao.r-1re7ezh.r-1qd0xha.r-n6v787.r-16dba41.r-1sf4r6n.r-bcqeeo.r-qvutc0")
trending_topics = [trend.text.strip() for trend in trends][:10]

# Preprocess the retrieved data using NLTK
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokens = word_tokenize(user_input.lower())
tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]

# Apply topic modeling techniques using Gensim (Latent Dirichlet Allocation)
dictionary = corpora.Dictionary([tokens])
corpus = [dictionary.doc2bow([token]) for token in tokens]
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)

# Generate a list of related topics based on the identified themes using WordNet
related_topics = []
for topic in lda_model.show_topics():
    words = topic[1].split("+")
    words = [word.split("*")[1].replace('"', '').strip() for word in words]
    for word in words:
        synsets = wordnet.synsets(word)
        for synset in synsets:
            related_topics += synset.lemma_names()

related_topics = list(set(related_topics))

# Present the generated topics to the user
print("Trending Topics:")
for topic in trending_topics:
    print(topic)

print("\nRelated Topics:")
for topic in related_topics:
    print(topic)
