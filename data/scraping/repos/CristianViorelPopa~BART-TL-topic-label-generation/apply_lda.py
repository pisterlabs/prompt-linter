import argparse
import pickle
from xml.dom import minidom

import numpy as np
from gensim.models import Phrases, LdaMulticore, HdpModel
from gensim.models.coherencemodel import CoherenceModel

from doc_preprocessing import preprocess_doc, remove_stopwords, remove_html_tags
from corpus_preprocessing import preprocess_corpus
import constants
from model_to_simple_csv import save_topic_words_simple_csv
from model_to_json import save_topic_words_json
from top_sentences import get_top_sentences_regular, get_top_sentences_cos10


def train_lda(corpus, dct, num_topics):
    # lda_model = HdpModel(corpus, dct, T=100)
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=dct,
                             random_state=0,
                             num_topics=num_topics,
                             passes=10,
                             # chunksize=1000,
                             batch=False,
                             alpha='asymmetric',
                             decay=0.5,
                             offset=64,
                             eta='auto',
                             eval_every=5,
                             iterations=100,
                             gamma_threshold=0.001,
                             per_word_topics=True,
                             workers=8)

    return lda_model


def main():
    parser = argparse.ArgumentParser(
        description='Script for applying LDA on a corpus'
    )
    parser.add_argument('--input-file', '-i', required=True, type=str,
                        help='XML file that contains the corpus LDA will be applied on')
    parser.add_argument('--output-prefix', '-o', required=True, type=str,
                        help='Prefix of the path where the files saving the LDA model and other '
                             'information will be stored')
    parser.add_argument('--topics-prefix', '-t', required=True, type=str,
                        help='Prefix of the path where the topics data will be stored')

    args = parser.parse_args()

    xmldoc = minidom.parse(args.input_file)
    itemlist = xmldoc.getElementsByTagName('row')

    docs = []
    for item in itemlist:
        doc = item.attributes['Body'].value
        doc = preprocess_doc(doc)

        docs.append(doc)

    print("Done pre-processing documents")

    dct, corpus, left_docs = preprocess_corpus(docs)
    print("Done pre-processing corpus")
    
    lda_model = train_lda(corpus, dct, constants.NUM_TOPICS)
    cm = CoherenceModel(model=lda_model, corpus=corpus, texts=docs, coherence='c_v')
    topics_coherences = np.array(cm.get_coherence_per_topic())

    topics_indices = np.where(topics_coherences > constants.COHERENCE_THRESHOLD)[0]

    # topics_top_sentences = get_top_sentences_regular(lda_model, corpus, constants.NUM_TOPICS,
    #                                                  constants.NUM_TOP_SENTENCES_RAW, itemlist,
    #                                                  left_docs, dct)
    topics_top_sentences = get_top_sentences_cos10(lda_model, corpus, constants.NUM_TOPICS,
                                                   constants.NUM_TOP_SENTENCES_RAW, itemlist,
                                                   left_docs, dct)

    # save the model, dict and corpus
    lda_model.save(args.output_prefix + '.model')
    dct.save(args.output_prefix + '.dct')
    with open(args.output_prefix + '_corpus.pickle', 'wb') as f:
        pickle.dump(corpus, f)
    # save topics to csv and json
    save_topic_words_simple_csv(lda_model, topics_indices, args.topics_prefix + '.csv')
    save_topic_words_json(lda_model, topics_indices, args.topics_prefix + '.json')

    # save the top raw sentences for each topic
    with open(args.topics_prefix + '_sentences_raw.txt', 'w') as sentences_file:
        for topic_idx in topics_indices:
            for k_idx in range(constants.NUM_TOP_SENTENCES_RAW):
                sentences_file.write(topics_top_sentences[topic_idx][k_idx] + '\n')
            sentences_file.write('\n')

    # save the top sentences for each topic
    with open(args.topics_prefix + '_sentences.txt', 'w') as sentences_file:
        for topic_idx in topics_indices:
            k_idx = 0
            sentences_written_count = 0
            while sentences_written_count < constants.NUM_TOP_SENTENCES:
                # "good" sentences should have a minimum length; this basically fails when the
                # entire paragraph is too short
                # count on the fact that there is redundancy when extracting sentences
                if len(topics_top_sentences[topic_idx][k_idx]) < \
                        constants.TOP_SENTENCE_LEN_LOWER_BOUND:
                    k_idx += 1
                    continue
                sentences_file.write(topics_top_sentences[topic_idx][k_idx] + '\n')
                k_idx += 1
                sentences_written_count += 1
            sentences_file.write('\n')


if __name__ == '__main__':
    main()
