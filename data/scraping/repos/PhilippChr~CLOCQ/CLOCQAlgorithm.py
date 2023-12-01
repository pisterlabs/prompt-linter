import json
import threading
import time

import spacy
import stanza
from networkx import json_graph
from rank_bm25 import BM25Okapi

from clocq.CoherenceGraph import CoherenceGraph, CoherenceScoreProcessor
from clocq.ConnectivityGraph import ConnectivityGraph, ConnectivityScoreProcessor
from clocq.StringLibrary import StringLibrary
from clocq.TopkProcessor import TopkProcessor
from clocq.Wikipedia2VecRelevance import Wikipedia2VecRelevance


class CLOCQAlgorithm:
    def __init__(
        self,
        kb,
        string_lib,
        method_name,
        ner,
        path_to_stopwords,
        path_to_wiki2vec_model,
        path_to_wikipedia_mappings,
        path_to_norm_cache=None,
        wikidata_search_cache=None,
        verbose=False,
    ):
        self.kb = kb
        self.method_name = method_name
        self.string_lib = string_lib
        self.wiki2vec = Wikipedia2VecRelevance(self.kb, path_to_stopwords, path_to_wiki2vec_model, path_to_wikipedia_mappings, path_to_norm_cache)
        self.wikidata_search_cache = wikidata_search_cache
        self.verbose = verbose

        # NER specific setting of nlp object
        self.ner = ner
        if ner == "stanza":
            self.nlp = stanza.Pipeline(lang="en", processors="tokenize,ner")
        elif ner == "spacy":
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = None

        # load stopwords for BM25
        with open(path_to_stopwords, "r") as file:
            self.stopwords = file.read().split("\n")

    def get_seach_space(self, question, parameters, include_labels=True, include_type=False):
        """Load parameters."""
        h_match = parameters["h_match"]
        h_rel = parameters["h_rel"]
        h_conn = parameters["h_conn"]
        h_coh = parameters["h_coh"]
        d = int(parameters["d"])
        k = parameters["k"]
        p_setting = parameters["p_setting"]
        bm25_limit = parameters["bm25_limit"]

        """ Get question words from question. """
        start = time.time()
        question_words = self.string_lib.get_question_words(question, self.ner, self.nlp)
        self._print_verbose(("Question: ", question))
        self._print_verbose(("Question words: ", question_words))
        self._print_verbose(("Time for question words: ", time.time() - start))

        """ Initialization. """
        itemlists = list()
        processes = list()
        topk_processors = list()

        """ Initialize question word top-k processors. """
        start = time.time()
        connectivity_graph = ConnectivityGraph()
        coherence_graph = CoherenceGraph(self.wiki2vec)
        for question_word_index, question_word in enumerate(question_words):
            topk_processor = TopkProcessor(
                self.kb,
                self.wiki2vec,
                connectivity_graph,
                coherence_graph,
                question_word_index,
                question_words,
                h_match=h_match,
                h_rel=h_rel,
                h_conn=h_conn,
                h_coh=h_coh,
                d=d,
                k=k,
                wikidata_search_cache=self.wikidata_search_cache,
                verbose=self.verbose,
            )
            topk_processors.append(topk_processor)
            t = threading.Thread(target=topk_processor.add_candidates_to_graph)
            processes.append(t)
            t.start()
        for process in processes:
            process.join()
        processes = list()
        self._print_verbose(("Time for initializing question word processor ", time.time() - start))

        """ Retrieve set of word-word pairs for coherence/connectivity computations. """
        question_word_pairs = self._get_question_word_pairs(question_words)

        """ Establish scores in connectivity graph. """
        start = time.time()
        for i, question_word_pair in enumerate(question_word_pairs):
            index1, index2 = question_word_pair
            connectivity_processor = ConnectivityScoreProcessor(self.kb, connectivity_graph,)
            candidates1 = topk_processors[index1].get_candidates()
            candidates2 = topk_processors[index2].get_candidates()
            t = threading.Thread(target=connectivity_processor.process, args=(candidates1, candidates2,),)
            processes.append(t)
            t.start()
        for process in processes:
            process.join()
        processes = list()
        self._print_verbose(("Time for establishing scores in connectivity graph", time.time() - start))

        """ Establish scores in coherence graph. """
        start = time.time()
        for question_word_pair in question_word_pairs:
            index1, index2 = question_word_pair
            coherence_processor = CoherenceScoreProcessor(self.wiki2vec, coherence_graph,)
            candidates1 = topk_processors[index1].get_candidates()
            candidates2 = topk_processors[index2].get_candidates()
            t = threading.Thread(target=coherence_processor.process, args=(candidates1, candidates2,),)
            processes.append(t)
            t.start()
        for process in processes:
            process.join()
        self._print_verbose(("Time for establishing scores in coherence graph", time.time() - start))
        processes = list()

        """ Compute top k candidates for each of the question words. """
        start = time.time()
        for topk_processor in topk_processors:
            t = threading.Thread(target=topk_processor.compute_top_k, args=(connectivity_graph, coherence_graph,),)
            processes.append(t)
            t.start()
        for process in processes:
            process.join()
        processes = list()
        self._print_verbose(("Time for top-k processors", time.time() - start))

        """ Fetch best KB items and extract search space. """
        start = time.time()
        kb_item_tuple = list()
        search_space = list()
        for j, topk_processor in enumerate(topk_processors):
            topklist = topk_processor.get_top_k()
            p = self._set_p(p_setting, topk_processor.k)  # set value of p
            for rank, item in enumerate(topklist):
                label = self.kb.item_to_single_label(item["id"])
                kb_item_tuple.append(
                    {
                        "item": {"id": item["id"], "label": label},
                        "question_word": question_words[j],
                        "score": item["score"],
                        "rank": rank,
                    }
                )
                search_space += self.kb.get_neighborhood(item["id"], p=p, include_labels=include_labels, include_type=include_type)
        self._print_verbose(("Time for retrieving search space", time.time() - start))

        """ OPTIONAL: prune search space using BM25 """
        if bm25_limit:
            search_space = self._bm25_pruning(question, search_space, bm25_limit)

        """ Return the search space and disambiguation results. """
        result = {"kb_item_tuple": kb_item_tuple, "search_space": search_space}
        return result

    def store_caches(self):
        """Store caches of the individual components."""
        self.wiki2vec.store_norm_cache()
        self.wikidata_search_cache.store_cache()
        self.string_lib.store_tagme_NER_cache()

    def print_results(self, string):
        """Print the string to the result file."""
        with open(self.method_name + ".txt", "a") as file:
            file.write(str(string) + "\n")

    def _print_verbose(self, string):
        """Print only if verbose is set."""
        if self.verbose:
            print(string)

    def _set_p(self, p_setting, k):
        """Set the value of p for the given p_setting."""
        if p_setting == "DYNAMIC1":
            p_value = 10 ** (6 - k)
        elif p_setting == "DYNAMIC2":
            p_value = 10 ** (5 - k)
        elif p_setting == "DYNAMIC3":
            p_value = 10 ** (5 - 0.5 * k)
        elif p_setting == "DYNAMIC4":
            p_value = 10 ** (4 - 0.5 * k)
        elif p_setting == "DYNAMIC5":
            p_value = 10 ** (k)
        elif p_setting == "DYNAMIC6":
            if k == 1:
                p_value = 10000
            elif k < 4:
                p_value = 1000
            else:
                p_value = 100
        else:
            p_value = int(p_setting)
        return p_value

    def _get_question_word_pairs(self, question_words):
        """
        Returns all pairs of question words (by index). Required to
        perform connectivity checks between items of all question word pairs.
        """
        workers = len(question_words)
        pairs_to_check = list()
        for i in range(workers):
            for j in range(workers):
                if j <= i:
                    continue
                pairs_to_check.append((i, j))
        return pairs_to_check

    def _bm25_pruning(self, question, search_space, bm25_limit):
        """
        Prune the search space using BM25 relevance: the question
        is the query and verbalized facts are the documents.
        A maximum of 'bm25_limit' facts are returned.
        """
        if len(search_space) > bm25_limit:
            fact_tuples = [(fact, self._verbalize_kb_fact(fact)) for fact in search_space]
            search_space = self._bm25_retrieve_top_facts(question, fact_tuples, bm25_limit)
        return search_space

    def _bm25_retrieve_top_facts(self, question, fact_tuples, bm25_limit):
        """
        Return the 'bm25_limit' facts using the verbalized collection.
        """
        tokenized_corpus = [self._tokenize(fact_tuple[1]) for fact_tuple in fact_tuples]
        mapping = {" ".join(self._tokenize(fact_tuple[1])): fact_tuple[0] for fact_tuple in fact_tuples}
        bm25 = BM25Okapi(tokenized_corpus)
        question_tok = self._tokenize(question)
        res = bm25.get_top_n(question_tok, tokenized_corpus, n=bm25_limit)
        # map from texts -> KB facts
        top_facts = list()
        for result in res:
            kb_fact_text = " ".join(result)
            kb_fact = mapping[kb_fact_text]
            top_facts.append(kb_fact)
        return top_facts

    def _verbalize_kb_fact(self, kb_fact):
        """
        Transform the n-tuple KB fact to text (used as document in BM25 pruning).
        """
        labels = [item["label"] for item in kb_fact]
        text = " ".join(labels)
        text = text.strip()
        return text

    def _tokenize(self, string):
        """Tokenize input string."""
        string = string.replace(",", "")
        # string = string.lower()
        return [word for word in string.split() if not word in self.stopwords]


if __name__ == "__main__":
    print("Test")
