import logging

import numpy as np
from gensim.models import CoherenceModel

from pimlico.core.modules.base import BaseModuleExecutor
from pimlico.datatypes.corpora import is_invalid_doc
from pimlico.utils.progress import get_progress_bar


class ModuleExecutor(BaseModuleExecutor):
    def execute(self):
        topics_top_words = self.info.get_input("topics_top_words")
        topic_word_lists = topics_top_words.topics_words
        corpus = self.info.get_input("corpus")
        vocab = self.info.get_input("vocab").get_data()

        coherence_measure = self.info.options["coherence"]
        window_size = self.info.options["window_size"]
        if coherence_measure == "u_mass":
            self.log.info("Using u_mass coherence measure")
        elif window_size is None:
            self.log.info("Using {} coherence measure with default window size".format(coherence_measure))
        else:
            self.log.info("Using {} coherence measure with window size of {}".format(coherence_measure, window_size))

        original_num_per_topic = [len(topic) for topic in topic_word_lists]
        self.log.info("Loaded words for topics: {}".format(", ".join(str(n) for n in original_num_per_topic)))

        # Prepare an iterator for the text data
        texts = CoherenceCorpus(corpus)
        # Check for words that aren't in the reference corpus
        self.log.info("Checking reference corpus for topic words")
        unseen_topic_words = set(sum(topic_word_lists, []))
        for words in texts:
            for word in words:
                if word in unseen_topic_words:
                    unseen_topic_words.remove(word)
            if len(unseen_topic_words) == 0:
                break
        if len(unseen_topic_words) == 0:
            self.log.info("All topic words were found in reference corpus")
            num_missing_words = [0 for __ in range(len(topic_word_lists))]
        else:
            self.log.warn("Some topic words were not found in reference corpus. Removing from evaluation: {}".format(
                ", ".join(list(unseen_topic_words))))
            topic_word_lists = [
                [word for word in topic if word not in unseen_topic_words] for topic in topic_word_lists
            ]
            self.log.info("Num words per topic after filtering: {}".format(
                ", ".join(str(len(topic)) for topic in topic_word_lists)))
            num_missing_words = [original - len(remaining) for (original, remaining) in
                                 zip(original_num_per_topic, topic_word_lists)]

        # Use Gensim's special coherence-computing architecture
        processes = self.processes if self.processes > 1 else 1
        if processes > 1:
            self.log.info("Using {} processes".format(processes))

        cm = CoherenceModel(topics=topic_word_lists,
                            texts=texts,
                            dictionary=vocab.as_gensim_dictionary(),
                            coherence=coherence_measure, processes=processes, window_size=window_size)

        # Don't output tonnes of logging
        from gensim.topic_coherence.text_analysis import logger as text_anal_logger
        text_anal_logger.setLevel(logging.ERROR)

        self.log.info("Computing coherence")
        coherences = np.array(cm.get_coherence_per_topic())
        mean_coh = np.mean(coherences)

        self.log.info("Topic coherences: {}".format(", ".join("{:.3f}".format(c) for c in coherences)))
        self.log.info("Mean coherence: {:.4f}".format(mean_coh))

        mean_coh_notnan = None
        if np.any(np.isnan(coherences)):
            if np.all(np.isnan(coherences)):
                self.log.info("All coherences were NaNs")
            else:
                self.log.info("Some coherences were NaNs")
                mean_coh_notnan = np.mean(coherences[np.where(~np.isnan(coherences))[0]])
                self.log.info("Mean coherence of non-NaN results: {:.4f}".format(mean_coh_notnan))

        with self.info.get_output_writer("mean_coherence") as writer:
            writer.write("coherence", mean_coh_notnan)

        # Output a report
        with self.info.get_output_writer("output") as writer:
            writer.write_file("""\
Num topics: {}
Vocab size: {:,}
Topic words for topics:
{}
Missing words per topic:
{}
Topic coherences:
{}
Mean coherence:
{:.4f}{}
        """.format(
                len(topic_word_lists), len(vocab),
                "\n".join(
                    "    {}: {}".format(topic_num,
                                        ", ".join(words)) for topic_num, words in enumerate(topic_word_lists)),
                ", ".join(str(n) for n in num_missing_words),
                "\n".join(
                    "    {}: {:.4f}".format(topic_num, coherence) for topic_num, coherence in enumerate(coherences)),
                mean_coh,
                "\n{:.4f}".format(mean_coh_notnan) if mean_coh_notnan is not None else ""
            ), text=True)


class CoherenceCorpus:
    def __init__(self, tokenized_corpus):
        self.tokenized_corpus = tokenized_corpus

    def __iter__(self):
        pbar = get_progress_bar(len(self.tokenized_corpus), title="Counting")
        for doc_name, doc in pbar(self.tokenized_corpus):
            if not is_invalid_doc(doc):
                yield [word for sent in doc.sentences for word in sent]
