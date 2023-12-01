from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel, LdaMulticore


# https://radimrehurek.com/gensim/models/ldamulticore.html
def model(
    bow_corpus: list[list[tuple[int, int]]],
    dict_corpus: corpora.Dictionary,
    passes: int,
    num_topics: int,
    alpha: Optional[Union[float, npt.ArrayLike, str]] = "symmetric",
    eta: Optional[Union[float, npt.ArrayLike, str]] = None,
    minimum_probability: float = 0.01,
    multicore: bool = True,
) -> LdaModel:
    model_class = LdaModel
    if multicore:
        model_class = LdaMulticore

    return model_class(
        corpus=bow_corpus,
        id2word=dict_corpus,
        passes=passes,
        num_topics=num_topics,
        alpha=alpha,
        eta=eta,
        minimum_probability=minimum_probability,
        random_state=1024,
    )


def coherence_score(
    model: LdaModel,
    text_corpus: list[list[str]],
    dict_corpus: corpora.Dictionary,
    coherence: str,
) -> float:
    cm = CoherenceModel(
        model=model, texts=text_corpus, dictionary=dict_corpus, coherence=coherence
    )
    return cm.get_coherence()


def add_topics_to_documents(
    model: LdaModel,
    bow_corpus: list[list[tuple[int, int]]],
    data: pd.DataFrame,
    num_topics: int,
) -> pd.DataFrame:
    topics_df = pd.DataFrame(columns=[f"topic:{idx}" for idx in range(num_topics)])

    for idx, doc in enumerate(bow_corpus):
        for topic in model.get_document_topics(doc):
            topics_df.loc[idx, f"topic:{topic[0]}"] = topic[1]
    return data.join(topics_df)


def get_number_of_topics(model: LdaModel) -> int:
    return model.get_topics().shape[0]
