import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tomotopy as tp
from hyfi import HyFI
from hyfi.task import BatchTask

from thematos.datasets import Corpus

from .config import LdaConfig, TrainConfig, TrainSummaryConfig, WordcloudConfig
from .prior import WordPrior
from .types import CoherenceMetrics, ModelSummary

logger = logging.getLogger(__name__)


class TopicModel(BatchTask):
    _config_group_ = "/model"
    _config_name_ = "topic"

    task_name: str = "topic"
    batch_name: str = "model"
    model_type: str = "BASE"
    wordprior: WordPrior = WordPrior()
    corpus: Corpus = Corpus()
    model_args: LdaConfig = LdaConfig()
    train_args: TrainConfig = TrainConfig()
    train_summary_args: TrainSummaryConfig = TrainSummaryConfig()
    wc_args: WordcloudConfig = WordcloudConfig()

    coherence_metric_list: List[str] = ["u_mass", "c_uci", "c_npmi", "c_v"]
    eval_coherence: bool = True
    set_wordprior: bool = False
    autosave: bool = True
    save_full: bool = True
    verbose: bool = False

    # internal attributes
    _model_: Optional[Any] = None
    _timestamp_: Optional[str] = None
    _coherence_metrics_: Optional[CoherenceMetrics] = None
    _model_summary_: Optional[ModelSummary] = None
    _ll_per_words_: List[Tuple[int, float]] = []
    _doc_ids_: List[Any] = None
    _doc_topic_dists_df_: Optional[pd.DataFrame] = None
    _topic_term_dists_df_: Optional[pd.DataFrame] = None

    @property
    def model_id(self) -> str:
        model_type = self.model_type.upper()
        margs = [model_type, self.batch_id, f"k({self.model_args.k})"]
        return "_".join(margs)

    @property
    def model(self):
        if self._model_ is None:
            raise ValueError("Model has not been trained yet.")
        return self._model_

    @property
    def coherence_metrics_dict(self) -> Dict:
        return self._coherence_metrics_.model_dump() if self._coherence_metrics_ else {}

    @property
    def model_summary_dict(self) -> Dict:
        return self._model_summary_.model_dump() if self._model_summary_ else {}

    @property
    def train_args_dict(self) -> Dict:
        return self.train_args.kwargs if self.train_args else {}

    @property
    def model_args_dict(self) -> Dict:
        return self.model_args.kwargs if self.model_args else {}

    @property
    def timestamp(self) -> str:
        if self._timestamp_ is None:
            raise ValueError("Model has not been trained yet.")
        return self._timestamp_

    @property
    def tp_corpus(self) -> tp.utils.Corpus:
        return self.corpus.corpus

    @property
    def doc_ids(self) -> List[Any]:
        if not self._doc_ids_:
            self._doc_ids_ = self.corpus.doc_ids
        return self._doc_ids_

    @property
    def ll_per_words(self) -> Optional[pd.DataFrame]:
        if not self._ll_per_words_:
            logger.warning("No log-likelihood per word found.")
            return None
        return pd.DataFrame(self._ll_per_words_, columns=["iter", "ll_per_word"])

    @property
    def doc_topic_dists(self) -> np.ndarray:
        dist_ = np.stack([doc.get_topic_dist() for doc in self.model.docs])
        dist_ /= dist_.sum(axis=1, keepdims=True)
        return dist_

    @property
    def topic_term_dists(self) -> np.ndarray:
        return np.stack(
            [self.model.get_topic_word_dist(k) for k in range(self.model.k)]
        )

    @property
    def doc_num_words(self) -> np.ndarray:
        return np.array([len(doc.words) for doc in self.model.docs])

    @property
    def used_vocab(self) -> List[str]:
        return list(self.model.used_vocabs)

    @property
    def term_frequency(self) -> np.ndarray:
        return self.model.used_vocab_freq

    @property
    def num_docs(self) -> int:
        return len(self.model.docs)

    @property
    def num_words(self) -> int:
        return self.model.num_words

    @property
    def num_total_vocabs(self) -> int:
        return len(self.model.vocabs) if self.model.vocabs else None

    @property
    def num_used_vocab(self) -> int:
        return len(self.model.used_vocabs)

    @property
    def num_topics(self) -> int:
        """Number of topics in the model

        It is the same as the number of columns in the document-topic distribution.
        """
        return self.model.k if self.model else len(self.doc_topic_dists[0])

    @property
    def topic_term_dists_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.topic_term_dists, columns=self.used_vocab)

    def get_doc_topic_dists_df(
        self,
        doc_topic_dists: Optional[np.ndarray],
        doc_ids: Optional[List[Any]],
    ) -> pd.DataFrame:
        if len(doc_topic_dists) != len(doc_ids):
            raise ValueError(
                f"Number of inferred topics ({len(doc_topic_dists)}) does not match with number of documents ({len(doc_ids)})"
            )
        columns = [f"topic{i}" for i in range(self.num_topics)]
        dists_df = pd.DataFrame(doc_topic_dists, columns=columns)
        doc_id_df = pd.DataFrame(doc_ids, columns=["id"])
        return pd.concat([doc_id_df, dists_df], axis=1)

    @property
    def model_file(self) -> str:
        f_ = f"{self.model_id}.mdl"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return str(self.model_dir / f_)

    @property
    def ll_per_words_file(self) -> str:
        f_ = f"{self.model_id}-ll_per_word.csv"
        return str(self.output_dir / f_)

    @property
    def ll_per_words_fig_file(self) -> str:
        f_ = f"{self.model_id}-ll_per_word.png"
        return str(self.output_dir / f_)

    @property
    def doc_topic_dists_file(self) -> str:
        f_ = f"{self.model_id}-doc_topic_dists.parquet"
        return str(self.output_dir / f_)

    @property
    def topic_term_dists_file(self) -> str:
        f_ = f"{self.model_id}-topic_term_dists.parquet"
        return str(self.output_dir / f_)

    @property
    def used_vocab_file(self) -> str:
        f_ = f"{self.model_id}-used_vocab.txt"
        return str(self.output_dir / f_)

    @property
    def train_summary_file(self) -> str:
        f_ = f"{self.model_id}-summary.txt"
        return str(self.output_dir / f_)

    @property
    def batch_model_summary_file(self) -> str:
        f_ = f"{self.batch_name}-summary.jsonl"
        return str(self.output_dir / f_)

    @property
    def ldavis_file(self) -> str:
        f_ = f"{self.model_id}-ldavis.html"
        return str(self.output_dir / f_)

    @property
    def topic_wordcloud_file_format(self) -> str:
        format_ = self.model_id + "-wordcloud_{topic_id:03d}.png"
        return str(self.output_dir / "wordclouds" / format_)

    def update_model_args(self, **kwargs) -> None:
        self.model_args = self.model_args.model_copy(update=kwargs)

    def _set_wordprior(self) -> None:
        if self.wordprior is None:
            logger.info("No word prior set.")
            return
        if self.verbose:
            logger.info("Set word prior with %s.", self.wordprior)
        for tno, words in self.wordprior.items():
            if self.verbose:
                logger.info("Set words %s to topic #%s as prior.", words, tno)
            for word in words:
                self.model.set_word_prior(
                    word,
                    [
                        self.wordprior.max_prior_weight
                        if i == int(tno)
                        else self.wordprior.min_prior_weight
                        for i in range(self.num_topics)
                    ],
                )

    def train(self) -> None:
        # reset model
        self._model_ = None

        if self.set_wordprior:
            self._set_wordprior()

        self._timestamp_ = datetime.now().strftime("%Y%m%d_%H%M%S")
        # train model
        self._train(self.model)
        # save model
        if self.eval_coherence:
            self.eval_coherence_value()
        if self.autosave:
            self.save()

    def _train(self, model: Any) -> None:
        raise NotImplementedError

    def eval_coherence_value(
        self,
    ):
        mdl = self.model
        coh_metrics = {}
        for metric in self.coherence_metric_list:
            coh = tp.coherence.Coherence(mdl, coherence=metric)
            average_coherence = coh.get_score()
            coh_metrics[metric] = average_coherence
            coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
            if self.verbose:
                logger.info("==== Coherence : %s ====", metric)
                logger.info("Average: %s", average_coherence)
                logger.info("Per Topic: %s", coherence_per_topic)
        self._coherence_metrics_ = CoherenceMetrics(**coh_metrics)

    def save(self) -> None:
        self.save_model()
        self.save_train_summary()
        self.save_ll_per_words()
        self.plot_ll_per_words()
        self.save_dists_data()
        self.save_topic_top_words()
        self.generate_wordclouds()
        self.save_ldavis()
        self.save_model_summary()
        self.save_config()

    def save_model(self) -> None:
        self.model.save(self.model_file, full=self.save_full)
        logger.info("Model saved to %s", self.model_file)

    def save_ll_per_words(self) -> None:
        HyFI.save_dataframes(
            self.ll_per_words, self.ll_per_words_file, verbose=self.verbose
        )

    def plot_ll_per_words(self) -> None:
        df_ll = self.ll_per_words
        ax = df_ll.plot(x="iter", y="ll_per_word", kind="line")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Log-likelihood per word")
        ax.invert_yaxis()
        ax.get_figure().savefig(self.ll_per_words_fig_file, dpi=300, transparent=False)
        logger.info(
            "Log-likelihood per word plot saved to %s", self.ll_per_words_fig_file
        )

    def save_train_summary(self) -> None:
        coh_values = self.coherence_metrics_dict
        original_stdout = sys.stdout
        Path(self.train_summary_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.train_summary_file, "w") as f:
            sys.stdout = f  # Change the standard output to the file.
            self.model.summary(**self.train_summary_args.kwargs)
            if coh_values:
                print("<Topic Coherence Scores>")
                for cm, cv in coh_values.items():
                    print(f"| {cm}: {cv}")
            sys.stdout = original_stdout  # Reset the standard output.

    def save_model_summary(self) -> None:
        self._model_summary_ = ModelSummary(
            timestamp=self.timestamp,
            model_id=self.model_id,
            model_type=self.model_type,
            num_docs=self.num_docs,
            num_words=self.num_words,
            num_total_vocabs=self.num_total_vocabs,
            num_used_vocabs=self.num_used_vocab,
            seed=self.seed,
            model_args=self.model_args_dict,
            train_args=self.train_args_dict,
            perplexity=self.model.perplexity,
            coherence=self.coherence_metrics_dict,
        )
        if not self.model_summary_dict:
            logger.warning("Model summary is not available.")
        HyFI.append_to_jsonl(
            self.model_summary_dict,
            self.batch_model_summary_file,
        )
        logger.info("Model summary saved to %s", self.batch_model_summary_file)

    def save_dists_data(self):
        doc_topic_dists_df = self.get_doc_topic_dists_df(
            self.doc_topic_dists, self.doc_ids
        )
        if self.verbose:
            logger.info("==== Document-Topic Distributions ====")
            logger.info(doc_topic_dists_df.tail())
        HyFI.save_dataframes(
            doc_topic_dists_df,
            self.doc_topic_dists_file,
            verbose=self.verbose,
        )
        if self.verbose:
            logger.info("==== Topic-Word Distributions ====")
            logger.info(self.topic_term_dists_df.tail())
        HyFI.save_dataframes(
            self.topic_term_dists_df,
            self.topic_term_dists_file,
            verbose=self.verbose,
        )
        HyFI.save_wordlist(
            self.used_vocab,
            self.used_vocab_file,
            verbose=self.verbose,
        )

    def load(
        self,
        batch_name: Optional[str] = None,
        batch_num: Optional[int] = None,
        filepath: Optional[Union[str, Path]] = None,
        **config_kwargs,
    ):
        super().load_config(
            batch_name=batch_name,
            batch_num=batch_num,
            filepath=filepath,
            **config_kwargs,
        )
        self._load_model()
        self._load_ll_per_words()
        self._load_dists_data()

    def _load_ll_per_words(self):
        ll_df = HyFI.load_dataframes(self.ll_per_words_file, verbose=self.verbose)
        self._ll_per_words_ = [(ll.iter, ll.ll_per_word) for ll in ll_df.itertuples()]

    def _load_dists_data(self):
        self._doc_topic_dists_df_ = HyFI.load_dataframes(
            self.doc_topic_dists_file, verbose=self.verbose
        )
        self._doc_ids_ = self._doc_topic_dists_df_["id"].values.tolist()
        self._topic_term_dists_df_ = HyFI.load_dataframes(
            self.topic_term_dists_file, verbose=self.verbose
        )

    def save_ldavis(self):
        try:
            import pyLDAvis  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning(
                "pyLDAvis is not installed. Please install it to save LDAvis."
            )
            return

        prepared_data = pyLDAvis.prepare(
            topic_term_dists=self.topic_term_dists,
            doc_topic_dists=self.doc_topic_dists,
            doc_lengths=self.doc_num_words,
            vocab=self.used_vocab,
            term_frequency=self.term_frequency,
            start_index=0,
            sort_topics=False,
        )
        pyLDAvis.save_html(prepared_data, self.ldavis_file)
        logger.info("LDAvis saved to %s", self.ldavis_file)

    def get_topic_words(
        self,
        topic_id: int,
        top_n: int = 10,
    ) -> Dict[str, float]:
        return dict(self.model.get_topic_words(topic_id, top_n=top_n))

    @property
    def topic_top_words_file(self) -> str:
        f_ = f"{self.model_id}-topic_top_words.txt"
        return str(self.output_dir / f_)

    @property
    def topic_top_words_dists_file(self) -> str:
        f_ = f"{self.model_id}-topic_top_words_dists.csv"
        return str(self.output_dir / f_)

    def save_topic_top_words(self, top_n: int = 50):
        # set of top words
        topic_top_words = []
        # tuple of (topic_id, word, freq) for each topic
        topic_top_words_dists = []
        for topic_id in range(self.num_topics):
            topic_words = self.get_topic_words(topic_id, top_n=top_n)
            topic_top_words.extend(topic_words.keys())
            topic_words_freq_tuple = [
                (topic_id, w, topic_words[w]) for w in topic_words
            ]
            topic_top_words_dists.extend(topic_words_freq_tuple)

        HyFI.save_wordlist(
            list(set(topic_top_words)),
            self.topic_top_words_file,
            verbose=self.verbose,
        )
        HyFI.save_dataframes(
            pd.DataFrame(topic_top_words_dists, columns=["topic_id", "word", "freq"]),
            self.topic_top_words_dists_file,
            verbose=self.verbose,
        )

    def generate_wordclouds(
        self,
    ):
        wc_args = self.wc_args
        wc = wc_args.wc
        images = []
        for topic_id in range(self.num_topics):
            output_file = self.topic_wordcloud_file_format.format(topic_id=topic_id)
            img = wc.generate_from_frequencies(
                self.get_topic_words(topic_id, top_n=wc_args.top_n),
                output_file=output_file,
                verbose=self.verbose,
            )
            images.append(img)

        if wc_args.make_collage:
            titles = wc_args.titles or [f"Topic {i}" for i in range(self.num_topics)]
            logger.info("Making wordcloud collage with titles: %s", titles)
            output_dir = self.output_dir / "wordcloud_collage"
            output_file_format = self.model_id + "_wordcloud_{page_num:02d}.png"
            HyFI.make_subplot_pages_from_images(
                images,
                num_images_per_page=wc_args.num_images_per_page,
                num_cols=wc_args.num_cols,
                num_rows=wc_args.num_rows,
                output_dir=output_dir,
                output_file_format=output_file_format,
                titles=titles,
                title_fontsize=wc_args.title_fontsize,
                title_color=wc_args.title_color,
                figsize=wc_args.figsize,
                width_multiple=wc_args.width_multiple,
                height_multiple=wc_args.height_multiple,
                dpi=wc_args.dpi,
                verbose=self.verbose,
            )

    @property
    def inferred_doc_topic_dists_filename(self) -> str:
        return f"{self.model_id}-inferred_doc_topic_dists.parquet"

    def infer(
        self,
        corpus: Corpus,
        output_file: Optional[Union[str, Path]] = None,
        iterations: int = 100,
        tolerance: float = -1,
        num_workers: int = 0,
        together: bool = False,
    ):
        inferred_corpus, ll = self.model.infer(
            corpus.corpus,
            iter=iterations,
            tolerance=tolerance,
            workers=num_workers,
            together=together,
        )
        logger.info("Number of documents inferred: %d", len(inferred_corpus))
        output_file = output_file or (
            self.output_dir / "inferred_topics" / self.inferred_doc_topic_dists_filename
        )

        doc_ids = corpus.doc_ids
        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in inferred_corpus])
        doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)

        doc_topic_dists_df = self.get_doc_topic_dists_df(doc_topic_dists, doc_ids)
        ll_df = pd.DataFrame({"log_likelihood": ll})
        doc_topic_dists_df = pd.concat([doc_topic_dists_df, ll_df], axis=1)
        if self.verbose:
            logger.info("Inferred topics:\n%s", doc_topic_dists_df.head())
        HyFI.save_dataframes(
            doc_topic_dists_df,
            output_file,
            verbose=self.verbose,
        )
        logger.info("Inferred topics saved to %s", output_file)
