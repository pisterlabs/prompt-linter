import getpass
import math
import os
import re
import string
import sys
from functools import reduce

import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pandas as pd
from nltk.corpus import stopwords
from stanza.server import CoreNLPClient
from AbstractClusterBERTUtility import AbstractClusterBERTUtility

nltk_path = os.path.join('/Scratch', getpass.getuser(), 'nltk_data')
nltk.download('stopwords', download_dir=nltk_path)
# Append NTLK data path
nltk.data.path.append(nltk_path)


# Helper function for keyword cluster
class KeywordExtractionUtility:
    stop_words = list(stopwords.words('english'))

    # Compute similarity score of keywords to the abstract
    # Ref:https://openai.com/blog/introducing-text-and-code-embeddings/
    @staticmethod
    def compute_similar_score_key_phrases_GPT(doc_vector, candidates, candidate_vectors):
        try:
            if len(candidates) == 0:
                return []

            # Encode cluster doc and keyword candidates into vectors for comparing the similarity
            # candidate_vectors = model.encode(candidates, convert_to_numpy=True)
            # Compute the distance of doc vector and each candidate vector
            distances = cosine_similarity(np.array([doc_vector]), np.array(candidate_vectors))[0].tolist()
            # Select top key phrases based on the distance score
            candidate_scores = list()
            # Get all the candidates sorted by similar score
            for candidate, distance in zip(candidates, distances):
                found = next((kp for kp in candidate_scores if kp['candidate'].lower() == candidate.lower()), None)
                if not found:
                    candidate_scores.append({'candidate': candidate, 'score': distance})
            # Sort the phrases by scores
            candidate_scores = sorted(candidate_scores, key=lambda k: k['score'], reverse=True)
            return candidate_scores
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Find top K key phrase similar to the paper
    # Ref: https://www.sbert.net/examples/applications/semantic-search/README.html
    @staticmethod
    def compute_similar_score_key_phrases(model, doc_text, candidates):
        try:
            if len(candidates) == 0:
                return []

            # Encode cluster doc and keyword candidates into vectors for comparing the similarity
            candidate_vectors = model.encode(candidates, convert_to_numpy=True)
            doc_vector = model.encode([doc_text], convert_to_numpy=True)  # Convert the numpy array
            # Compute the distance of doc vector and each candidate vector
            distances = cosine_similarity(doc_vector, candidate_vectors)[0].tolist()
            # Select top key phrases based on the distance score
            candidate_scores = list()
            # Get all the candidates sorted by similar score
            for candidate, distance in zip(candidates, distances):
                found = next((kp for kp in candidate_scores if kp['key-phrase'].lower() == candidate.lower()), None)
                if not found:
                    candidate_scores.append({'key-phrase': candidate, 'score': distance})
            # Sort the phrases by scores
            candidate_scores = sorted(candidate_scores, key=lambda k: k['score'], reverse=True)
            return candidate_scores
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    @staticmethod
    # Generate Collocation using regular expression patterns
    def generate_collocation_candidates(doc_text, client):
        try:
            candidates = list()
            ann = client.annotate(doc_text)
            # Extract n_gram from each sentence
            for sentence in ann.sentence:
                pos_tags = list()
                # sentence_tokens = list()
                for token in sentence.token:
                    pos_tags.append(token.originalText + "_" + token.pos)
                    # sentence_tokens.append(token.originalText)
                sentence_tagged_text = ' '.join(pos_tags)
                sentence_tagged_text = sentence_tagged_text.replace(" -_HYPH ", " ")  # Remove the hype
                # Use the regular expression to obtain n_gram
                # Patterns: (1) JJ* plus NN and NN+
                #           (2) JJ and JJ NN plus NN*
                #           (3) JJ+ plus NN plus NN*
                #           (4) JJ* plus NN plus NN+
                sentence_words = list()
                pattern = r'((\w+_JJ\s+)*(\w+_NN[P]*[S]*\s*(\'s_POS)*\s+)(\s*\,_\,\s*)*(and_CC\s+)(\w+_NN[P]*[S]*\s*(\'s_POS)*\s+){1,})' \
                          r'|((\w+_JJ\s+)(and_CC\s+)(\w+_JJ\s+)(\w+_NN[P]*[S]*\s*(\'s_POS)*\s+){1,})' \
                          r'|((\w+_JJ\s+){1,}(\w+_NN[P]*[S]*\s*(\'s_POS)*\s*){1,})' \
                          r'|((\w+_JJ\s+)*(\w+_NN[P]*[S]*\s*(\'s_POS)*\s+){2,})'
                matches = re.finditer(pattern, sentence_tagged_text)
                for match_obj in matches:
                    try:
                        n_gram = match_obj.group(0)
                        n_gram = n_gram.replace(" 's_POS", "'s")
                        n_gram = n_gram.replace(" ,_,", "")
                        n_gram = n_gram.replace("_CC", "")
                        n_gram = n_gram.replace("_JJ", "")
                        n_gram = n_gram.replace("_NNPS", "")
                        n_gram = n_gram.replace("_NNP", "")
                        n_gram = n_gram.replace("_NNS", "")
                        n_gram = n_gram.replace("_NN", "")
                        n_gram = n_gram.replace("_VBN", "")
                        n_gram = n_gram.replace("_VBG", "")
                        n_gram = n_gram.strip()
                        sentence_words.append(n_gram)
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                        sys.exit(-1)
                # print(sentence_words)
                for word in sentence_words:
                    found = next((cw for cw in candidates if cw.lower() == word.lower()), None)
                    if not found:
                        candidates.append(word)
            return candidates
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)

    # Get candidate words by using POS patterns for each doc in
    @staticmethod
    def generate_tfidf_terms(cluster_docs, folder):
        # Generate n-gram of a text and avoid stop
        def _generate_single_word_candidates(_doc_text, _client):
            def _is_qualified(_word):  # _n_gram is a list of tuple (word, tuple)
                try:
                    # Check if all words are not stop word or punctuation or non-words
                    if bool(re.search(r'\d|[^\w]', _word.lower())) or _word.lower() in string.punctuation or \
                            _word.lower() in KeywordExtractionUtility.stop_words:
                        return False
                    # n-gram is qualified
                    return True
                except Exception as err:
                    print("Error occurred! {err}".format(err=err))

                ann = client.annotate(_doc_text)

            candidates = list()
            ann = _client.annotate(_doc_text)
            # Extract n_gram from each sentence
            for sentence in ann.sentence:
                try:
                    sentence_tokens = list()
                    for token in sentence.token:
                        sentence_tokens.append(token.originalText)
                    sentence_candidates = list()
                    # Filter out not qualified n_grams that contain stopwords or the word is not alpha_numeric
                    for token in sentence_tokens:
                        if _is_qualified(token):
                            sentence_candidates.append(token)  # Add token to a string
                    candidates = candidates + sentence_candidates
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
            return candidates

        # Create frequency matrix to track the frequencies of a n-gram in
        def _create_frequency_matrix(_docs, _client):
            # Vectorized the clustered doc text and Keep the Word case unchanged
            frequency_matrix = []
            for doc in _docs:
                _doc_id = doc['DocId']  # doc id
                doc_text = AbstractClusterBERTUtility.preprocess_text(doc['Abstract'])
                freq_table = {}
                candidates = _generate_single_word_candidates(doc_text, _client)
                for candidate in candidates:
                    term = candidate.lower()
                    if candidate.isupper():
                        term = candidate
                    if term in freq_table:
                        freq_table[term] += 1
                    else:
                        freq_table[term] = 1
                frequency_matrix.append({'doc_id': _doc_id, 'freq_table': freq_table})
            return frequency_matrix

        # Compute TF score
        def _compute_tf_matrix(_freq_matrix):
            _tf_matrix = {}
            # Compute tf score for each cluster (doc) in the corpus
            for _row in _freq_matrix:
                _doc_id = _row['doc_id']  # Doc id is the cluster no
                _freq_table = _row['freq_table']  # Store the frequencies of each word in the doc
                _tf_table = {}  # TF score of each word (such as 1, 2, 3-gram) in the doc
                _total_terms_in_doc = reduce(lambda total, f: total + f, _freq_table.values(), 0)
                # Adjusted for total number of words in doc
                for _term, _freq in _freq_table.items():
                    # frequency of a word in doc / total number of words in doc
                    _tf_table[_term] = _freq / _total_terms_in_doc
                _tf_matrix[_doc_id] = _tf_table
            return _tf_matrix

        # Collect the table to store the mapping between word to a list of clusters
        def _create_occ_per_term(_freq_matrix):
            _occ_table = {}  # Store the mapping between a word and its doc ids
            for _row in _freq_matrix:
                _doc_id = _row['doc_id']  # Doc id is the cluster no
                _freq_table = _row['freq_table']  # Store the frequencies of each word in the doc
                for _term, _count in _freq_table.items():
                    if _term in _occ_table:  # Add the table if the word appears in the doc
                        _occ_table[_term].add(_doc_id)
                    else:
                        _occ_table[_term] = {_doc_id}
            return _occ_table

        # Compute IDF scores
        def _compute_idf_matrix(_freq_matrix, _occ_per_term):
            _total_cluster = len(_freq_matrix)  # Total number of clusters in the corpus
            _idf_matrix = {}  # Store idf scores for each doc
            for _row in _freq_matrix:
                _doc_id = _row['doc_id']  # Doc id is the cluster no
                _freq_table = _row['freq_table']  # Store the frequencies of each word in the doc
                _idf_table = {}
                for _term in _freq_table.keys():
                    _counts = len(_occ_per_term[_term])  # Number of clusters the word appears
                    _idf_table[_term] = math.log10(_total_cluster / float(_counts))
                _idf_matrix[_doc_id] = _idf_table  # Idf table stores each word's idf scores
            return _idf_matrix

        # Compute tf-idf score matrix
        def _compute_tf_idf_matrix(_tf_matrix, _idf_matrix, _freq_matrix, _occ_per_term):
            _tf_idf_matrix = list()
            # Compute tf-idf score for each cluster
            for _doc_id, _tf_table in _tf_matrix.items():
                # Compute tf-idf score of each word in the cluster
                _idf_table = _idf_matrix[_doc_id]  # idf table stores idf scores of the doc (doc_id)
                # Get freq table of the cluster
                _freq_table = next(f for f in _freq_matrix if f['doc_id'] == _doc_id)['freq_table']
                _tf_idf_list = []
                for _term, _tf_score in _tf_table.items():  # key is word, value is tf score
                    try:
                        _idf_score = _idf_table[_term]  # Get idf score of the word
                        _freq = _freq_table[_term]  # Get the frequencies of the word in doc_id
                        _doc_ids = sorted(list(_occ_per_term[_term]))  # Get the clusters that the word appears
                        _score = float(_tf_score * _idf_score)
                        _tf_idf_list.append({'term': _term, 'score': _score, 'freq': _freq, 'doc_ids': _doc_ids})
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                # Sort tf_idf_list by tf-idf score
                _term_list = sorted(_tf_idf_list, key=lambda t: t['score'], reverse=True)
                _tf_idf_matrix.append({'doc_id': _doc_id, 'terms': _term_list})
                # Write the selected output to csv files
                if _doc_id in [206, 325, 523]:
                    # Write to a list
                    _term_df = pd.DataFrame(_term_list, columns=['term', 'score', 'freq', 'doc_ids'])
                    # Write the topics results to csv
                    _term_df.to_csv(os.path.join(folder, 'TF-IDF_doc_terms_' + str(_doc_id) + '.csv'), encoding='utf-8',
                                    index=False)
            return _tf_idf_matrix

        try:
            # Use Stanford CoreNLP to tokenize the text
            with CoreNLPClient(
                    annotators=['tokenize', 'ssplit'],
                    timeout=30000,
                    be_quiet=True,
                    memory='6G') as client:
                # 2. Create the Frequency matrix of the words in each document (a cluster of articles)
                freq_matrix = _create_frequency_matrix(cluster_docs, client)
                # # 3. Compute Term Frequency (TF) and generate a matrix
                # # Term frequency (TF) is the frequency of a word in a document divided by total number of words in the document.
                tf_matrix = _compute_tf_matrix(freq_matrix)
                # # 4. Create the table to map the word to a list of documents
                occ_per_term = _create_occ_per_term(freq_matrix)
                # # 5. Compute IDF (how common or rare a word is) and output the results as a matrix
                idf_matrix = _compute_idf_matrix(freq_matrix, occ_per_term)
                # # Compute tf-idf matrix
                terms_list = _compute_tf_idf_matrix(tf_matrix, idf_matrix, freq_matrix, occ_per_term)
                return terms_list  # Return a list of dicts
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)

    # Get a list of unique key phrases from all papers
    @staticmethod
    def sort_candidates_by_similar_score(phrase_scores):
        try:
            # Sort 'phrase list'
            sorted_phrase_list = sorted(phrase_scores, key=lambda p: p['score'], reverse=True)
            unique_key_phrases = list()
            for key_phrase in sorted_phrase_list:
                # find if key phrase exist in all key phrase list
                found = next((kp for kp in unique_key_phrases
                              if kp['key-phrase'].lower() == key_phrase['key-phrase'].lower()), None)
                if not found:
                    unique_key_phrases.append(key_phrase)
                else:
                    print("Duplicated: " + found['key-phrase'])

            # Return unique key phrases
            return unique_key_phrases
        except Exception as _err:
            print("Error occurred! {err}".format(err=_err))

    # Maximal Marginal Relevance minimizes redundancy and maximizes the diversity of results
    # Ref: https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea
    @staticmethod
    def re_rank_phrases_by_maximal_margin_relevance(doc_vector, candidates, candidate_vectors, diversity=0.5, top_k=20):
        try:
            top_n = min(top_k, len(candidates))
            # doc_vector = model.encode([doc_text], convert_to_numpy=True)
            # phrase_vectors = model.encode(phrase_candidates, show_progress_bar=True, convert_to_numpy=True)

            # Extract similarity within words, and between words and the document
            candidate_doc_similarity = cosine_similarity(np.array(candidate_vectors), np.array([doc_vector]))
            candidate_similarity = cosine_similarity(np.array(candidate_vectors), np.array(candidate_vectors))

            # Pick up the most similar phrase
            most_similar_index = np.argmax(candidate_doc_similarity)
            # Initialize candidates and already choose the best keyword/key phrases
            keyword_idx = [most_similar_index]
            top_keywords = [{'keyword': (candidates[most_similar_index]),
                             'score': candidate_doc_similarity[most_similar_index][0]}]
            # Get all the remaining index
            candidate_indexes = list(filter(lambda idx: idx != most_similar_index, range(len(candidates))))
            # Add the other candidate phrase
            for i in range(0, top_n - 1):
                # Get similarities between doc and candidates
                candidate_similarities = candidate_doc_similarity[candidate_indexes, :]
                # Get similarity between candidates and a set of extracted key phrases
                target_similarities = candidate_similarity[candidate_indexes][:, keyword_idx]
                # Calculate MMR
                mmr_scores = (1 - diversity) * candidate_similarities - diversity * np.max(target_similarities,
                                                                                           axis=1).reshape(-1, 1)
                mmr_idx = candidate_indexes[np.argmax(mmr_scores)]

                # Update keywords & candidates
                top_keywords.append(
                    {'keyword': candidates[mmr_idx], 'score': candidate_doc_similarity[mmr_idx][0]})
                keyword_idx.append(mmr_idx)
                # Remove the phrase at mmr_idx from candidate
                candidate_indexes = list(filter(lambda idx: idx != mmr_idx, candidate_indexes))
            return top_keywords
        except Exception as err:
            print("Error occurred! {err}".format(err=err))
            sys.exit(-1)
