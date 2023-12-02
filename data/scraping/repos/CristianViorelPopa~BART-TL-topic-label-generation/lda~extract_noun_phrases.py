import argparse
import re
import string
import pickle
from collections import Counter
from xml.dom import minidom

import numpy as np
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import spacy

import constants
from doc_preprocessing import preprocess_doc, remove_stopwords, remove_html_tags

nlp = spacy.load("en")
determinants = ['a', 'an', 'the', 'what', 'its', 'her', 'his', 'my', 'your', 'their', 'this', 'these', 'that', 'those', 'both', 'any', 'such', 'only', 'other', 'same', 'most', 'many', 'few', 'more', 'another', 'all', 'but', 'also', 'though', 'however', 'and', 'too', 'or']
artifacts = ['&amp;c', '\n', 'nbsp']


def preprocess_noun_phrases(noun_phrase):
    noun_phrase = noun_phrase.lower()
    for artifact in artifacts:
        noun_phrase = noun_phrase.replace(artifact, '')
    noun_phrase = process_np(noun_phrase)
    noun_phrase = noun_phrase.replace('_', ' ')
    noun_phrase = noun_phrase.replace('\n', ' ').replace('\t', ' ')
    noun_phrase = re.sub(' +', ' ', noun_phrase)
    words = noun_phrase.strip().split(' ')
    prev_words = ''
    while words != prev_words:
        prev_words = words
        if len(words) == 0:
            break
        if words[0] in determinants or words[0] in string.punctuation:
            words = words[1:]
    noun_phrase = ' '.join(words).strip()
    return noun_phrase


def process_np(np_):
    exclude = string.punctuation + '‘“”'
    include = ['\'', ',', '-', '_']
    exclude = list(filter(lambda c: c not in include, exclude))
    replace = [('’', '\'')]
    for c in exclude:
        np_ = np_.replace(c, '')
    for replace_from, replace_to in replace:
        np_ = np_.replace(replace_from, replace_to)
    np_ = np_.replace('_', ' ')
    np_ = re.sub(' +', ' ', np_)
    np_ = np_.strip().replace(' ', '_')
    return np_


def get_noun_phrases(lda_model, corpus, num_topics, topics_indices, original_texts, left_docs, dct):
    original_texts = np.array([text.attributes['Body'].value for text in original_texts])
    left_docs = np.array(left_docs)

    # get the doc-topics distribution and populate a matrix for it
    docs_topics = lda_model.get_document_topics(corpus)
    topics_distribution = np.zeros((len(corpus), num_topics))
    for doc_idx in range(len(corpus)):
        for topic_idx, prob in docs_topics[doc_idx]:
            topics_distribution[doc_idx][topic_idx] = prob
    # sort the doc-topics distribution, in order to get the top K docs for every topic; shape of the
    # array will be K x num_topics
    sorted_topics_distribution = np.argsort(topics_distribution, axis=0)

    all_noun_phrases = []
    all_noun_phrases_counter = Counter()
    for topic_idx in topics_indices:
        topic_cand_docs = sorted_topics_distribution[:, topic_idx]
        topic_cand_docs = topic_cand_docs[-constants.TOPIC_DOC_SIMILARITY_THRESHOLD:]
        topic_cand_docs = left_docs[topic_cand_docs]

        all_topic_noun_phrases = set()
        for text in original_texts[topic_cand_docs]:
            # find noun phrases in text
            doc = nlp(remove_html_tags(text.strip()))
            noun_phrases = [np_.text for nc in doc.noun_chunks for np_ in [nc, doc[nc.root.left_edge.i:nc.root.right_edge.i + 1]]]
            noun_phrases = [preprocess_noun_phrases(noun_phrase) for noun_phrase in noun_phrases]
            noun_phrases = list(filter(lambda np_: constants.MIN_NOUN_PHRASE_LEN <=
                                                   len(np_.split(' ')) <=
                                                   constants.MAX_NOUN_PHRASE_LEN, noun_phrases))

            all_topic_noun_phrases.update(set(noun_phrases))
            all_noun_phrases_counter += Counter(noun_phrases)

        # sort the noun phrases by topic similarity
        all_topic_noun_phrases = np.array(list(all_topic_noun_phrases))
        noun_phrases_topics = [dict(lda_model[dct.doc2bow(noun_phrase.split(' '))])
                               for noun_phrase in all_topic_noun_phrases]
        noun_phrases_topic_prob = [noun_phrase_topics[topic_idx] if topic_idx in noun_phrase_topics
                                   else 0.0 for noun_phrase_topics in noun_phrases_topics]
        ranked_noun_phrases = np.argsort(np.array(noun_phrases_topic_prob))[::-1]
        all_noun_phrases.append(all_topic_noun_phrases[ranked_noun_phrases])

        print('Done topic ' + str(topic_idx))

    # filter noun phrases based on number of appearances
    valid_noun_phrases = {np_ for np_, count in all_noun_phrases_counter.items()
                          if count >= constants.NOUN_PHRASES_OCCURRENCES_MIN_THRESHOLD}
    all_noun_phrases = [[np_ for np_ in topic_nps if np_ in valid_noun_phrases]
                        for topic_nps in all_noun_phrases]

    print(len(valid_noun_phrases))

    # select the top N noun phrases
    all_noun_phrases = [list(dict.fromkeys(topic_nps[:constants.NUM_NOUN_PHRASES_PER_TOPIC]))
                        for topic_nps in all_noun_phrases]

    return all_noun_phrases


def main():
    parser = argparse.ArgumentParser(
        description='Script for extracting noun phrases for topics and saving them for future use'
    )
    parser.add_argument('--lda-path', '-l', required=True, type=str,
                        help='Path to the saved LDA model (X.model)')
    parser.add_argument('--dict-path', '-d', required=True, type=str,
                        help='Path to the saved training dictionary (X.dct)')
    parser.add_argument('--corpus-path', '-c', required=True, type=str,
                        help='Path to the pickled gensim-processed corpus (X_corpus.pickle)')
    parser.add_argument('--input-file', '-i', required=True, type=str,
                        help='XML file that contains the corpus LDA was initially trained on (or '
                             'the one the noun phrases will be extracted from)')
    parser.add_argument('--output-prefix', '-o', required=True, type=str,
                        help='Prefix of the path where the noun phrases will be stored')

    args = parser.parse_args()

    lda_model = LdaMulticore.load(args.lda_path)
    dct = corpora.Dictionary.load(args.dict_path)
    with open(args.corpus_path, 'rb') as f:
        corpus = pickle.load(f)

    xmldoc = minidom.parse(args.input_file)
    itemlist = xmldoc.getElementsByTagName('row')

    docs = []
    for item in itemlist:
        doc = item.attributes['Body'].value
        doc = preprocess_doc(doc)

        docs.append(doc)

    left_docs = [idx for idx, doc in enumerate(docs) if len(doc) > constants.NUM_WORDS_THRESHOLD]

    cm = CoherenceModel(model=lda_model, corpus=corpus, texts=docs, coherence='c_v')
    topics_coherences = np.array(cm.get_coherence_per_topic())

    topics_indices = np.where(topics_coherences > constants.COHERENCE_THRESHOLD)[0]

    print('Done preamble.')

    noun_phrases = get_noun_phrases(lda_model, corpus, constants.NUM_TOPICS, topics_indices,
                                    itemlist, left_docs, dct)

    # write noun phrases to file
    with open(args.output_prefix + '_noun_phrases.txt', 'w') as noun_phrases_file:
        for topic_noun_phrases in noun_phrases:
            noun_phrases_file.write(' '.join([process_np(noun_phrase.strip().replace(' ', '_'))
                                              for noun_phrase in topic_noun_phrases]) + '\n')


if __name__ == '__main__':
    main()
