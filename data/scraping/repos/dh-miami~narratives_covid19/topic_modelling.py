from os.path import isfile
from os import makedirs
import re
from random import sample
from json import dump, load

import pandas as pd
import stanza
import spacy
from tqdm import tqdm
import gensim
from gensim import corpora
from gensim.models import Phrases
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
import pyLDAvis.gensim

AVAILABLE_LANGS = ['es', 'en']

def remove_emojis(tweet):
    return re.sub(r':[^: ]+?:', '', tweet)

def check_valid_lang(lang):
    if lang not in AVAILABLE_LANGS:
        raise Exception(f'There is no stopwords file for {lang}.')    

def read_tweets_csv(path):
    '''Reads a csv from coveet.py and returns a list of tweets.'''
    data = pd.read_csv(path)
    return list(data.text.dropna())

def read_stopwords(lang):
    '''Reads stopword file, returns a list of stopwords.'''
    check_valid_lang(lang)
    with open(f'../stopwords/stopwords_{lang}.txt', 'r') as fi:
        stop_words = fi.read()
    return stop_words.split()
    
def read_emoji_labels():
    with open('emojinames.parsed.extras.txt') as fi:
        emoji_labels = fi.read()
    return emoji_labels.split()

def make_stanza_tweets(lang, tweets, processors='tokenize, pos, lemma'):
    '''Receives a list of tweets and a language code. 
    Processors is passed to stanza.Pipeline.
    Returns a list of stanza Documents'''
    check_valid_lang(lang)
    nlp_stanza = stanza.Pipeline(lang, processors=processors)
    return [nlp_stanza(tweet) for tweet in tqdm(tweets, desc='Tokenizing, applying POS and lemmatizing tweets')]

def clean(tweet, stop_words):
    clean_tweet=[]
    for token in tweet.split():
        if not token.isnumeric():
            if token not in stop_words:
                clean_tweet.append(token)
    return ' '.join(clean_tweet)

def remove_empty_tweets(tweets):
    return [tw for tw in tweets if tw]

def count_words_in_tweet(tweet):
    return len(tweet.split())

def remove_one_word_tweets(tweets):
    return [tw for tw in tweets if count_words_in_tweet(tw) > 1]

def clean_stanza(stanza_word, stop_words):
    '''Receives a Stanza dictionary'''
    not_a_number = not stanza_word.text.isnumeric()
    not_a_stopword = (stanza_word.text not in stop_words) and (stanza_word.lemma not in stop_words)
    #not_an_unidentified_emoji = 'emoji_not_identified' not in stanza_word.text ## novamas
    #not_an_emoji = stanza_word.text not in emojinames_list ## novamas
    
    return not_a_number and not_a_stopword #and not_an_unidentified_emoji and not_an_emoji

def remove_NE_tag(txt):
    return txt.replace("NE__", "").replace("ne__", "")

def dump_processed_tweets_as_json(tweets, filepath):
    """tweets should be a list of lists of strings."""
    with open(filepath, "w") as f:
        dump(tweets, f)

def make_lemmas_list_nouns_list(stanza_tweets, stop_words):
    '''Receives stanza tweets and returns a list of lemmatized tweets and a list of lemmatized tweets (only nouns).
    Each tweet is a list.'''
    lemmas_tweets = [] # all the tweets, without stopwords, lemmatized
    nouns_tweets = [] # all the tweets, without stopwords, lemmatized, nouns only

    for tweet_doc in stanza_tweets:
        this_tweet_all_lemmas = []
        this_tweet_noun_lemmas = []

        for sent in tweet_doc.sentences:
            for stanza_word in sent.words:
                if not clean_stanza(stanza_word, stop_words):
                    continue
                    
                lemma = remove_NE_tag(stanza_word.lemma)
                text = remove_NE_tag(stanza_word.text)

                this_tweet_all_lemmas.append(lemma)
                
                # Warning: stanza_word.text keeps uppercase characters but lemma does not!
                if "NE__" in stanza_word.text:
                    this_tweet_noun_lemmas.append(text)
                elif stanza_word.upos == "NOUN":
                    this_tweet_noun_lemmas.append(lemma)
                elif stanza_word.upos == "PROPN":
                    this_tweet_noun_lemmas.append(text)
      
        lemmas_tweets.append(this_tweet_all_lemmas)
        nouns_tweets.append(this_tweet_noun_lemmas)
        
    return lemmas_tweets, nouns_tweets


# Generate bigrams (only the ones that appear 20 times or more, all types of POS).
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#pre-process-and-vectorize-the-documents

def add_ngrams_to_tweets(lemmas_tweets, nouns_tweets, verbose=True):
    '''Generates ngrams of each tweet in lemmas_tweets (if ngram appears 20 times or more in tweets)
    and adds them to the corresponding tweets in nouns_tweets. Returns a copy of nouns_tweets modified.'''
    bigram = Phrases(lemmas_tweets, min_count=20)
    nouns_tweets = nouns_tweets.copy()
    
    for lemma_tweet, noun_tweet in zip(lemmas_tweets, nouns_tweets):
        tweet_ngrams = [token for token in bigram[lemma_tweet] if '_' in token ]
        noun_tweet += tweet_ngrams
        if tweet_ngrams and verbose:
            print(f'Adding ngrams: {tweet_ngrams}')

    return nouns_tweets


def make_dictionary_and_matrix(tweets):
    '''Makes gensim dictionary and doc_term_matrix objects for LDA model training.
    Returns (dictionary, doc_term_matrix)'''
    dictionary = corpora.Dictionary(tweets)
    doc_term_matrix = [dictionary.doc2bow(tweet) for tweet in tweets]
    return (dictionary, doc_term_matrix)

def train_LDA_model(ntopics, dictionary, doc_term_matrix, output_path=None):
    '''Receives topic number '''
    print(f"Training LDA model with {ntopics} topics")
    model = LdaModel(
        doc_term_matrix,
        num_topics=ntopics,
        id2word=dictionary,
        passes=20,
        eval_every=1,
        iterations=50
    )
    if output_path:
        print(f"Saving in {output_path}")
        model.save(output_path)
    return model

def load_LDA_model(model_path):
    return LdaModel.load(model_path)

def make_model_path(models_dir, corpus_label, model_label, ntopics):
    model_path = f"{models_dir}/{corpus_label}-{model_label}-ntopics{ntopics:02}"
    return model_path

def train_several_LDA_models(tweets, topic_numbers_to_try, corpus_label, model_label, models_dir, overwrite=False):
    models = {}
    
    makedirs(models_dir, exist_ok=True)

    for ntopics in topic_numbers_to_try:
        output_path = make_model_path(models_dir, corpus_label, model_label, ntopics)

        if not isfile(output_path) or overwrite:
            dictionary, doc_term_matrix = make_dictionary_and_matrix(tweets)
            train_LDA_model(ntopics, dictionary, doc_term_matrix, output_path=output_path)
        else:
            print(f"Already trained for {ntopics} topics")

        models[ntopics] = load_LDA_model(output_path)
        
    return models

def calculate_topic_coherence(models, tweets, measures=["c_npmi", "c_uci", "u_mass", "c_v"], verbose=True):
    '''models should be a dictionary of ntopics as keys, LDA models as values.
    Returns pandas.DataFrame of scores'''
    scores = []
    dictionary = corpora.Dictionary(tweets)
    
    models_iterator = models.items()
    if verbose:
        models_iterator = tqdm(models_iterator, desc='Calculating model coherence: ')
    
    for ntopics, model in models_iterator:
        scoring = {"ntopics": ntopics}
        for measure in measures:
            # Based on: https://radimrehurek.com/gensim/models/coherencemodel.html
            # The 'texts' must be the same preprocessing -if any- we used to generate the model we are loading!  
            cm = CoherenceModel(model=model, dictionary=dictionary, coherence=measure, texts=tweets)
            scoring[measure] = cm.get_coherence()

        scores.append(scoring)
    
    return pd.DataFrame(scores)

def plot_scores(corpus_label, model_label, models_dir, scores):
    with plt.style.context("bmh"):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        for i, measure in enumerate(["c_npmi", "c_uci", "u_mass", "c_v"]):
            ax = axes.flat[i]
            ax.plot(scores.ntopics, scores[measure], "-o")
            ax.set_title(f"Coherence measured with: {measure}\n{corpus_label}\n{model_label}")
            ax.set_xlabel("Number of Topics")
            ax.set_ylabel("Coherence Score")

        plt.tight_layout()
        plt.savefig(f'{models_dir}/{corpus_label}-{model_label}-coherence.png')
        plt.show()
        

def plot_LDA_topics(model, tweets, output_path=None, show=True, notebook=True):
    dictionary, doc_term_matrix = make_dictionary_and_matrix(tweets)
    
    if notebook:
        pyLDAvis.enable_notebook()
        
    data = pyLDAvis.gensim.prepare(model, doc_term_matrix, dictionary)
        
    if output_path:
        print(f"Saving topic visualization to: {output_path}")
        pyLDAvis.save_html(data, output_path)
    
    if show:
        return data

def load_processed_tweets_from_json(path):
    with open(path) as f:
        return load(f)

def make_tweets_with_tagged_named_entities(spacy_tweets):
    modified_tweets = []
    for spacy_tweet in tqdm(spacy_tweets):
        modified_tweet = str(spacy_tweet)

        for entity in spacy_tweet.ents:
            entity_words = str(entity).split()
            entity_merged = "_".join(entity_words)
            modified_tweet = modified_tweet.replace(str(entity), f"NE__{entity_merged}")

        modified_tweets.append(modified_tweet)
    return modified_tweets

def infer_lang_from_corpus_label(label):
    # Example label: dhcovid_2020-7-11_2020-7-17_es_mx
    return label.split("_")[-2]

SPACY_MODELS = {
    'es': 'es_core_news_sm',
    'en': 'en_core_web_lg'
}

def modelize(path, sample_n=None, overwrite=False):
    # Assumes these previous steps:
    # $ python -m spacy download en_core_web_lg
    # $ python -m spacy download es_core_news_sm
    # stanza.download('en')
    # stanza.download('es')
    
    corpus_label = path.replace('.csv', '')
    processed_tweets_path = path.replace(".csv", ".processed-tweets.json")

    if not isfile(processed_tweets_path):
        lang = infer_lang_from_corpus_label(corpus_label)
        tweets = read_tweets_csv(path)

        if sample_n:
            tweets = sample(tweets, min(sample_n, len(tweets)))

        print(f"[{corpus_label}] Preprocessing")
        tweets = [remove_emojis(tweet).strip() for tweet in tweets]
        tweets = [tweet for tweet in tweets if tweet]
        stopwords = read_stopwords(lang)
        clean_tweets = []
        for tweet in tweets:
            clean_tweets.append(clean(tweet, stopwords))

        # Hack needed to avoid weird Stanza behaviour with Spanish.
        # For instance, "peinandose" breaks the "pos" processor, but
        # "peinandose peinandose" does not break it!
        clean_tweets = remove_one_word_tweets(clean_tweets)

        print(f"[{corpus_label}] Identify Named Entities")
        spacy_nlp = spacy.load(SPACY_MODELS[lang])
        nlp_spacy_tweets = [spacy_nlp(tweet, disable=["tagger", "parser"]) for tweet in tqdm(clean_tweets)]

        ner_tweets = make_tweets_with_tagged_named_entities(nlp_spacy_tweets)
        ner_tweets = remove_empty_tweets(ner_tweets)

        print(f"{corpus_label}] POS and lemmatization")
        stanza_tweets = make_stanza_tweets(lang, ner_tweets)

        print(f"[{corpus_label}] Build list of lemmas and nouns")
        lemmas_tweets, nouns_tweets = make_lemmas_list_nouns_list(stanza_tweets, stopwords)

        nouns_tweets_with_bigrams = add_ngrams_to_tweets(lemmas_tweets, nouns_tweets, verbose=False)

        print(f"[{corpus_label}] Dump processed tweets as JSON")
        dump_processed_tweets_as_json(nouns_tweets_with_bigrams, processed_tweets_path)
    else:
        print(f"[{corpus_label}] Load processed tweets from JSON")
        nouns_tweets_with_bigrams = load_processed_tweets_from_json(processed_tweets_path)
    
    print(f"[{corpus_label}] Unsupervised learning")
    model_label = 'lemma_2gram_LDA'
    models_dir = "../../outputs/topic_modelling/gensim_LDA_models"

    models = train_several_LDA_models(
        tweets=nouns_tweets_with_bigrams,
        topic_numbers_to_try=range(3, 13),
        corpus_label=corpus_label,
        model_label=model_label,
        models_dir=models_dir,
        overwrite=overwrite
    )