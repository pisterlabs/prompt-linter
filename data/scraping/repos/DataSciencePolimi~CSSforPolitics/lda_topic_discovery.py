import sys, os
sys.path.append("/home/brambilla/users/emre/CSSforPolitics/")
from gensim.models import CoherenceModel, LdaModel
from gensim import corpora, similarities
import warnings
import logging as logger
warnings.simplefilter("ignore", DeprecationWarning)
from util import utils, ml_utils
from topic_modeling import preprocess_corpus as preprocessor
import traceback
import pandas as pd

data_path = "/home/ubuntu/users/emre/CSSforPolitics/topic_modeling/data/"
#data_path = "F:/tmp/"
topic_number = 20
chunksize = 10200
epochs = 20
model_eval_every = 4
max_iterations = 200
alpha = 'auto'
beta = topic_number / 2000
lda_model_save_enabled = False
visual_enabled = True

logger.basicConfig(filename=data_path + 'topic.log', format="%(asctime)s:%(levelname)s:%(message)s", level=logger.INFO)


def topic_discovery():
    try:
        number_of_files_splitted_periods = 1
        for i in range(1, number_of_files_splitted_periods + 4):

            # Load texts
            filename = data_path + 'p' + str(i) + '.csv'
            #filename = "F:/tmp/test"
            df = utils.read_file(filename, "~", names=['ID', 'datetime', 'text'])
            texts = df["text"].tolist()

            dictionary, corpus, texts = create_dictionary_corpus(texts)

            lm = LdaModel(corpus=corpus, id2word=dictionary,
                          num_topics=topic_number,
                          chunksize=chunksize,
                          passes=epochs,
                          eval_every=model_eval_every,
                          iterations=max_iterations,
                          alpha=alpha,
                          eta=beta,
                          )
            output_visual_file_name = filename + ".vis"
            ml_utils.evaluate_lda_results(corpus, dictionary, texts, lm, topic_number,
                                          output_visual_file_name, visual_enabled)
            combined_topic_id_file_name = filename + "_topic_out.csv"
            combine_lda_results_with_lda_output(corpus, lm, df, combined_topic_id_file_name)

            if lda_model_save_enabled:
                lm.save(data_path + 'LDA_model_' + str(i) + '.lda')

    except Exception as ex:
        logger.error(str(ex))
        logger.info(traceback.format_exc())


def create_dictionary_corpus(texts):
    print('Preprocessing texts...')
    texts, bigram_mod = preprocessor.preprocess(texts, use_bigrams=True)

    print('Creating dictionary...')
    dictionary = corpora.Dictionary(texts)
    # Filter out words that occur less than 10 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    # dictionary.save(data_path + 'dictionary_'+str(i)+'.dict')  # store the dictionary, for future reference

    print('Serializing corpus...')
    # corpora.MmCorpus.serialize(data_path + 'corpus_'+str(i)+'.mm', [dictionary.doc2bow(t) for t in texts])
    corpus = [dictionary.doc2bow(t) for t in texts]
    return dictionary, corpus, texts


def combine_lda_results_with_lda_output(corpus, lda_model, df, file_out):
    topic_ids = []
    try:
        for bow in corpus:
            topics = lda_model.get_document_topics(bow)
            max_prob_topic = max(topics, key=lambda item: item[1])
            topic_ids.append(max_prob_topic[0])
        logger.info("original document nb of rows: " + str(df.shape[0]))
        logger.info("new topic calculation nb of rows" + str(len(topic_ids)))
        if (df.shape[0] != len(topic_ids)):
            logger.error("FATAL ERROR caused by data mismatch: len other cols: " + str(
                df.shape[0]) + " len new topic cols:" + len(
                topic_ids))
            return None;

        df["topic_id"] = pd.Series(topic_ids)
        df.to_csv(file_out, index=False)
        logger.info(
            "saved succesfully each tweet assigned with the topic having the highest probability into the file: " + file_out)
    except Exception as ex:
        logger.info("could not saved into a file")
        logger.error(str(ex))
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    topic_discovery()

