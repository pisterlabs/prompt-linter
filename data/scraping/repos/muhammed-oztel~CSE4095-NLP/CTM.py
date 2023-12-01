import nltk
import pandas as pd
import numpy as np
from contextualized_topic_models.models.kitty_classifier import Kitty
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from contextualized_topic_models.evaluation.measures import CoherenceNPMI, TopicDiversity
import pyLDAvis as vis
from collections import Counter

# function to check if a string is an integer
def isint(word):
    try:
        int(word)
        return True
    except:
        return False

# function to creating additional turkish stopwords
def create_stop_words(number):
    additional_stop_words = ['dır', 'nev', 'nın']
    for word in uni_grams.most_common():
        if word[1] > number:
            additional_stop_words.append(word[0])
        elif len(word[0]) <= 2 or isint(word[0]):
            additional_stop_words.append(word[0])
    
    return additional_stop_words

# loading dataset
data = pd.read_csv("cleaned_data_for_ctm.csv").dropna()
embeddings = np.load("text_embeddings.npy")

turkish_stopwords = nltk.corpus.stopwords.words('turkish')

# data preprocessing
documents = [line.strip() for line in data['text'].values]
sp = WhiteSpacePreprocessingStopwords(documents, turkish_stopwords, vocabulary_size=5000)
preprocessed_documents, unpreprocessed_documents, vocab, retained_indices = sp.preprocess()

# finding the most common unigrams
uni_grams = Counter()
for text in preprocessed_documents:
    uni_grams.update(text.split())

# that removes the most common unigrams used more than 200 times
additional_stop_words = create_stop_words(200)
stop_words = turkish_stopwords + additional_stop_words

# training model
kt = Kitty()
kt.train(data['text'].values.tolist(), custom_embeddings=embeddings, 
        stopwords_list=stop_words, topics=10, 
        hidden_sizes=(200, 200), n_words=5000)

# saving lda_vis figure
lda_vis_data = kt.get_ldavis_data_format()
ctm_pd = vis.prepare(**lda_vis_data)
vis.save_html(ctm_pd, "lda_vis_figure_200.html")

# fingding and saving probabilities
probs = kt.ctm.get_thetas(kt.ctm.train_data)
np.save("probs.npy", probs)

# saving model
kt.save("ctm_model.pkl")

# coherenceNPMI and topic diversity scores
texts = [i.split() for i in preprocessed_documents]
topic_list = kt.ctm.get_topic_lists(20)
npmi = CoherenceNPMI(texts=texts, topics=topic_list)
td = TopicDiversity(topic_list)

print(npmi.score(), td.score(topk=20))