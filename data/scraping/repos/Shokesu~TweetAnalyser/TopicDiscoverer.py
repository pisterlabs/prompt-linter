import pandas as pd
from pprint import pprint
import sys, traceback
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import logging as logger
import warnings
import utils
import globals

warnings.filterwarnings("ignore", category=DeprecationWarning)


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def convert_file_to_df(filename):
    logger.info("started to read document")
    try:
        ids = []
        logger.info("filename: " + str(filename))
        df = pd.read_csv(filename, names=['ID', 'datetime', 'tweet'])
        logger.info(str(df.shape[0]) + " tweets are taken into memory")
    except Exception as ex:
        logger.info("Something bad happened: %s", ex)
        logger.info(ex)

    logger.info("completed reading document")
    return df


def remove_unwanted_words_from_df(df):
    texts_unwanted_eliminated = []
    for text in df['tweet'].values.tolist():
        new_text = text.replace("brexit", "")
        new_text = new_text.replace("eu", "")
        new_text = new_text.replace("  ", " ")
        texts_unwanted_eliminated.append(new_text)
    return texts_unwanted_eliminated


def main():
    try:

        logger.info("Started Topic discovery operations")
        logger.basicConfig(level="INFO", filename=globals.WINDOWS_LOG_PATH, format="%(asctime)s %(message)s")

        # data = df.tweet.values.tolist()
        logger.info("started LDA related operations")

        filename_read = "F:/tmp/p1_test.csv"
        df = convert_file_to_df(filename_read)

        texts_unwanted_eliminated = remove_unwanted_words_from_df(df)
        data_words = list(sent_to_words(texts_unwanted_eliminated))

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)

        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words, bigram_mod)
        logger.info(data_words_bigrams[:1])

        # python3 -m spacy download en

        #logger.info("spacy - lemmatization started")
        #nlp = spacy.load('en', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        #data_lemmatized = lemmatization(nlp, data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        #logger.info("spacy - lemmatization completed")

        # Create Dictionary
        logger.info("creating corpora")

        # id2word = corpora.Dictionary(data_lemmatized)
        id2word = corpora.Dictionary(data_words_bigrams)

        # Create Corpus
        texts = data_words_bigrams

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        # df["bow"]=corpus

        # View
        logger.info(corpus[:1])

        logger.info([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

        logger.info("building LDA model")
        topic_cnt = 20
        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=topic_cnt,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
        top_topics = lda_model.top_topics(corpus=corpus)

        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        avg_topic_coherence = sum([t[1] for t in top_topics]) / topic_cnt
        logger.info("Average topic coherence: " + str(avg_topic_coherence))

        logger.info("top topics ordered by coherence score")
        logger.info(str(top_topics))

        # Print the Keyword in the 10 topics
        logger.info("topics: " + str(lda_model.print_topics()))

        ids = []
        datetimes =[]
        topic_ids = []
        # topic_words = []
        counter_index = 0

        for bow in corpus:
            topics = lda_model.get_document_topics(bow)
            topic_counter = 0
            max_prob = 0
            max_prob_topic = None
            for topic in topics:
                prob = topic[1]
                if max_prob < prob:
                    max_prob = prob
                    max_prob_topic = topic
                else:
                    break

            topic_ids.append(max_prob_topic[0])
            tweet_id = df.iloc[counter_index]['ID']
            datetime = df.iloc[counter_index]['datetime']
            ids.append(tweet_id)
            datetimes.append(datetime)
            counter_index += 1

        if len(ids) != len(topic_ids):
            logger.error("FATAL ERROR caused by data mismatch: len other cols: " + len(ids) + " len new topic cols:" + len(topic_ids))
            exit(-1)

        newdf = pd.DataFrame(
            {
                'ID': ids,
                'datetime': datetimes,
                'topic_id': topic_ids
            })
        if (df.shape[0] != newdf.shape[0]):
            logger.info("FATAL ERROR caused by data mismatch: the number of lines are not matching in input and output data collections")
            sys.exit(-1)

        logger.info("completed operations")

        file_output = filename_read + "_topic_out.csv"
        newdf.to_csv(file_output, index=False)

        logger.info("saved succesfully into a file")

        # Print the Keyword in the 10 topics
        logger.info(lda_model.print_topics())
        logger.info("topics: : " + str(lda_model.print_topics()))

        # mallet operations... comment line because takes error in ubuntu
        # logger.info("ldamallet topics: " + ldamallet.show_topics(formatted=False))
        # Compute Coherence Score
        # coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word,coherence='c_v')
        # coherence_ldamallet = coherence_model_ldamallet.get_coherence()
        # logger.info('\nldamallet Coherence Score: ', coherence_ldamallet)

        # Compute Perplexity
        logger.info("Perplexity: %s", lda_model.log_perplexity(corpus))

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        logger.info("Coherence Score: " + str(coherence_lda))

        # Visualize the topics
        # pyLDAvis.enable_notebook()
        visual_enabled = True
        if visual_enabled:
            vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
            lda_file = filename_read + "_LDA_Visualization.html"
            pyLDAvis.save_html(vis, lda_file)


    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
        logger.error(ex)
        logger.error("Something bad happened: %s", ex)

    logger.info("Completed everything. Program is being terminated")


if __name__ == "__main__":
    main()
