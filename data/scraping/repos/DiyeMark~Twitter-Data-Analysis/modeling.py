import spacy
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import gensim
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

best_model = ""
best_model_name = ""
best_score = 0


def sentiment_analysis_topic_modeling():
    clean_tweets = pd.read_csv('data/clean_tweet_data.csv')
    clean_tweets = clean_tweets.fillna("")

    # Sentiment Analysis
    sentiment_analysis_tweet_data = clean_tweets.copy(deep=True)
    sentiment_analysis_tweet_data.drop(
        sentiment_analysis_tweet_data[sentiment_analysis_tweet_data['sentiment'] == -1].index, inplace=True)
    sentiment_analysis_tweet_data.reset_index(drop=True, inplace=True)
    tweet_train = sentiment_analysis_tweet_data.iloc[:10622, ]
    tweet_test = sentiment_analysis_tweet_data.iloc[10623:, ]

    unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
    unigram_vectorizer.fit(tweet_train['clean_text'].values)

    X_train_unigram = unigram_vectorizer.transform(
        tweet_train['clean_text'].values)

    unigram_tf_idf_transformer = TfidfTransformer()
    unigram_tf_idf_transformer.fit(X_train_unigram)

    X_train_unigram_tf_idf = unigram_tf_idf_transformer.transform(
        X_train_unigram)

    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
    bigram_vectorizer.fit(tweet_train['clean_text'].values)

    X_train_bigram = bigram_vectorizer.transform(
        tweet_train['clean_text'].values)

    bigram_tf_idf_transformer = TfidfTransformer()
    bigram_tf_idf_transformer.fit(X_train_bigram)

    X_train_bigram_tf_idf = bigram_tf_idf_transformer.transform(X_train_bigram)

    def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, train_size=0.75, stratify=y
        )

        clf = SGDClassifier()
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        valid_score = clf.score(X_valid, y_valid)

        global best_model
        global best_model_name
        global best_score

        if (valid_score > best_score):
            best_model = clf
            best_model_name = title
            best_score = valid_score

        print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')

    y_train = tweet_train['sentiment'].values

    train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
    train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
    train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
    train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')

    print(
        f'The best Model is {best_model_name} with a Validation score of: {round(best_score, 2)}')

    def run_test_using_model(best_model: SGDClassifier, model_type: str):
        unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
        unigram_vectorizer.fit(tweet_test['clean_text'].values)
        X_test_unigram = unigram_vectorizer.transform(
            tweet_test['clean_text'].values)

        bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
        bigram_vectorizer.fit(tweet_test['clean_text'].values)
        X_test_bigram = bigram_vectorizer.transform(
            tweet_test['clean_text'].values)

        y_test = tweet_test['sentiment'].values

        if (model_type == "Unigram Counts"):
            X_test = X_test_unigram

        elif (model_type == "Unigram Tf-Idf"):
            unigram_tf_idf_transformer = TfidfTransformer()
            unigram_tf_idf_transformer.fit(X_test_unigram)
            X_test_unigram_tf_idf = unigram_tf_idf_transformer.transform(
                X_test_unigram)

            X_test = X_test_unigram_tf_idf

        elif (model_type == "Bigram Counts"):
            X_test = X_test_bigram

        else:
            bigram_tf_idf_transformer = TfidfTransformer()
            bigram_tf_idf_transformer.fit(X_test_bigram)

            X_test_bigram_tf_idf = bigram_tf_idf_transformer.transform(
                X_test_bigram)
            X_test = X_test_bigram_tf_idf

        return best_model.score(X_test, y_test)

    sgd = joblib.dump(best_model, './trained_models/newsentimentSGDmodel.jl')

    # Topic Modeling
    topic_model_data = clean_tweets.copy(deep=True)

    def get_hastags_words_list():
        hashtagList = []
        for hashtags in topic_model_data.hashtags:
            if (hashtags != ""):
                hashtagList += hashtags.split(',')

        return hashtagList

    hashtag = get_hastags_words_list()

    data = [
        word for sentence in topic_model_data.clean_text for word in sentence.split(' ')]

    data_words = data + hashtag
    data_words = [word for word in data_words if word != '']

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append(
                [token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=[
        'NOUN', 'ADJ', 'VERB', 'ADV'])

    data_lemmatized = [word for word in data_lemmatized if word != []]

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=20,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    doc_lda = lda_model[corpus]

    perplexity_score = lda_model.log_perplexity(corpus)
    print('\nPerplexity: ', perplexity_score)

    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    joblib.dump(lda_model, './trained_models/newtopicLDAmodel.jl')

    description = {'sentiment_analysis': {'name': best_model_name, 'score': best_score},
                   'topic_modeling': {'perplexity_score': perplexity_score, 'coherence_score': coherence_lda}}
    joblib.dump(description, './trained_models/newtrainedModelsData.jl')

    print('Sentiment and Topic Model Trained and Successfully Saved.!!!')


if __name__ == '__main__':
    sentiment_analysis_topic_modeling()
