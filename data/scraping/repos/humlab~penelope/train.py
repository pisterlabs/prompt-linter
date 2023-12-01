from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from ...interfaces import InferredModel, TrainingCorpus
from . import coherence, convert, options


def train(
    train_corpus: TrainingCorpus,
    method: str,
    engine_args: Dict[str, Any],
    **kwargs: Dict[str, Any],
) -> InferredModel:
    """Computes a topic model using Gensim as engine.

    Parameters
    ----------
    train_corpus : TrainingCorpus
        A container for the training data (terms or DTM, id2word, document_index)
    method : str
        The method to use (see `options` module for mappings)
    engine_args : Dict[str, Any]
        Generic topic modelling options that are translated to algorithm-specific options (see `options` module for translation)
    kwargs : Dict[str,Any], optional
        Additional vectorize options:
            `tfidf_weighing` if TF-IDF weiging should be applied, ony valid when terms/id2word are specified, by default False

    Returns
    -------
    InferredModel
        train_corpus        Training corpus data (updated)
        model               The engine specific topic model
        options:
            perplexity_score    Computed perplexity scores
            coherence_score     Computed coherence scores
            engine_options       Passed engine options (not the interpreted algorithm specific options)
            extra_options       Any other compute option passed as a kwarg
    """

    corpus, dictionary = convert.TranslateCorpus().translate(train_corpus.corpus, id2token=train_corpus.id2token)

    if kwargs.get('tfidf_weighing', False):
        logger.warning("TF-IDF weighing of effective corpus has been disabled")
        # tfidf_model = TfidfModel(corpus)
        # corpus = [tfidf_model[d] for d in corpus]

    train_corpus.corpus = corpus
    if train_corpus.token2id is None:
        train_corpus.token2id = dictionary.token2id

    engine_spec: options.EngineSpec = options.get_engine_specification(engine_key=method)
    model: Any = engine_spec.engine(
        **engine_spec.get_options(
            corpus=train_corpus.corpus,
            id2word=train_corpus.id2token,
            engine_args=engine_args,
        )
    )

    # FIXME: These metrics must be computed on a held-out corpus - not the training corpus
    # perplexity_score = (
    #     None
    #     if not hasattr(model, 'log_perplexity')
    #     else 2 ** model.log_perplexity(train_corpus.corpus, len(train_corpus.corpus))
    # )
    perplexity_score = 0

    coherence_score = coherence.compute_score(train_corpus.id2token, model, train_corpus.corpus)

    return InferredModel(
        topic_model=model,
        id2token=train_corpus.id2token,
        options=dict(
            method=method,
            perplexity_score=perplexity_score,
            coherence_score=coherence_score,
            engine_options=engine_args,
            extra_options=kwargs,
        ),
    )
