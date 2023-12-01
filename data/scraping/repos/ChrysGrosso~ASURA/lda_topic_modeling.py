import gensim
from gensim.models import CoherenceModel
import spacy
from textblob import TextBlob

# Script inspired by Alfred Tang - Latent Topics Modeling for Steam User reviews https: // medium.com / @ alfredtangsw / steamvox - sujet - modélisation - sentiment - analyse - d83a88d3003a"
#Chrys Grosso

# To prepare the data for the LDA, we will make n-grams, then lemmatise using spacy to postag
def spacy_lemma(bow, allowed_postags=None):
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'VERB', 'ADJ']
    nlp = spacy.load("en_core_web_sm") #after installing spacy, to download en_core_web_sm, tape directly in the terminal python -m spacy download en_core_web_sm
    lemma_doc = nlp(" ".join(bow))
    lemma_text = [token.text if '_' in token.text else token.lemma_ if token.pos_ in allowed_postags else '' for token in lemma_doc]
    return lemma_text

# Build the trigram models
def make_trigrams(df_clean):
    clean_reviews = df_clean['clean_reviews']
    bigram = gensim.models.Phrases(clean_reviews, min_count=5, threshold=10)
    trigram = gensim.models.Phrases(bigram[list(clean_reviews)], threshold=10)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    texts = clean_reviews
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

#LDA wit sentiment tone

def get_sentiment_score(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score
#FYI : TextBlob more accurate than Vader

def classify_sentiments(sentiment_lst):
    #TextBlob slightly overestimates positive dimension and underestimate negative dimension #light fix #positive score > 0.1, negative score <0
    pos_lst = [x for x in sentiment_lst if x > 0.1]
    #positive scores
    neutral_lst = [x for x in sentiment_lst if x >= 0 and x <= 0.1]
    #TextBlob
    neg_lst = [x for x in sentiment_lst if x < 0] # negative scores
    total_len = len(sentiment_lst)
    pos_percentage = len(pos_lst) / total_len
    neutral_percentage = len(neutral_lst) / total_len
    neg_percentage = len(neg_lst) / total_len
    return pos_percentage, neutral_percentage, neg_percentage











