import pandas as pd
import re
import stopwords.stopwords as s
import lda.loader as loader
import lda.cleaner as cleaner
import lda.tokenizer as tok
import pyLDAvis.gensim
import os
import spacy
from spacy.tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.corpora import Dictionary
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from gensim.models.coherencemodel import CoherenceModel
from sklearn.model_selection import GridSearchCV
from gensim.models.ldamulticore import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS as SW
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
from datetime import date


def start(num_topics, kind):
    data = loader.load_data(kind)
    df = pd.DataFrame(data)
    cleaner.clean(df)

    nlps = {
        'it': spacy.load('it_core_news_lg'),
        'en': spacy.load('en_core_web_lg'),
        'fr': spacy.load('fr'),
        'de': spacy.load('de')
    }

    tokenizers = {
        'it': Tokenizer(nlps['it'].vocab),
        'en': Tokenizer(nlps['en'].vocab),
        'fr': Tokenizer(nlps['fr'].vocab),
        'de': Tokenizer(nlps['de'].vocab)
    }

    # Customize stop words by adding to the default list
    stop_words = []
    stop_words += nlps['it'].Defaults.stop_words
    stop_words += nlps['en'].Defaults.stop_words
    stop_words += nlps['fr'].Defaults.stop_words
    stop_words += nlps['de'].Defaults.stop_words
    stop_words += s.ALL_STOPWORDS
    stop_words = set(stop_words)

    # ALL_STOP_WORDS = spacy + gensim + wordcloud
    ALL_STOP_WORDS = stop_words.union(SW).union(stopwords)

    cleaner.remove_stopwords(df, tokenizers, ALL_STOP_WORDS)
    cleaner.lemmas(df, nlps)

    tok.tokenize_text(df)

    # Create a id2word dictionary
    id2word = Dictionary(df['lemma_tokens'])
    print(len(id2word))

    # Filtering Extremes
    id2word.filter_extremes(no_below=2, no_above=.99)
    print(len(id2word))

    # Creating a corpus object
    corpus = [id2word.doc2bow(d) for d in df['lemma_tokens']]

    # Instantiating a Base LDA model
    base_model = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=id2word, workers=12, passes=5)

    # Filtering for words
    words = [re.findall(r'"([^"]*)"', t[1]) for t in base_model.print_topics()]

    # Create Topics
    topics = [' '.join(t[0:10]) for t in words]


    # Getting the topics
    for id, t in enumerate(topics):
        print(f"------ Topic {id} ------")
        print(t, end="\n\n")

    # Compute Perplexity
    # a measure of how good the model is. lower the better
    base_perplexity = base_model.log_perplexity(corpus)
    print('\nPerplexity: ', base_perplexity)

    # Compute Coherence Score
    coherence_model = CoherenceModel(model=base_model, texts=df['lemma_tokens'],
                                     dictionary=id2word, coherence='c_v')
    coherence_lda_model_base = coherence_model.get_coherence()
    print('\nCoherence Score: ', coherence_lda_model_base)

    lda_display = pyLDAvis.gensim.prepare(base_model, corpus, id2word)
    d = pyLDAvis.display(lda_display)

    today = date.today()
    directory_path = f"/home/marco/Scrivania/tirocinio-unicredit/lda-html/{kind}/{today}/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    f = open(f"/home/marco/Scrivania/tirocinio-unicredit/lda-html/{kind}/{today}/{num_topics}.html", 'w')
    f.write(d.data)
    f.close()

    vectorizer = CountVectorizer()
    data_vectorized = vectorizer.fit_transform(df['lemmas_back_to_text'])

    # Define Search Param
    search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

    # Init the Model
    lda = LatentDirichletAllocation()

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)

    # Do the Grid Search
    model.fit(data_vectorized)
    GridSearchCV(cv=None, error_score='raise',
                 estimator=LatentDirichletAllocation(batch_size=128,
                                                     doc_topic_prior=None,
                                                     evaluate_every=-1,
                                                     learning_decay=0.7,
                                                     learning_method=None,
                                                     learning_offset=10.0,
                                                     max_doc_update_iter=100,
                                                     max_iter=10,
                                                     mean_change_tol=0.001,
                                                     n_components=10,
                                                     n_jobs=1,
                                                     perp_tol=0.1,
                                                     random_state=None,
                                                     topic_word_prior=None,
                                                     total_samples=1000000.0,
                                                     verbose=0),
                 iid=True, n_jobs=1,
                 param_grid={'n_topics': [10, 15, 20, 30],
                             'learning_decay': [0.5, 0.7, 0.9]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
                 scoring=None, verbose=0)

    # Best Model
    best_lda_model = model.best_estimator_

    # Model Parameters
    print("Best Model's Params: ", model.best_params_)

    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)

    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

if __name__ == '__main__':
    # kinds =
    # 'all' for all tweets,
    # 'it' (all the company and tweet with ISIN IT)
    # 'authoritative' for the authoritative twitter account
    start(30, 'authoritative')
