import os
import warnings
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import openai
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from tqdm import tqdm

from ..model import Model
from ..utils import (get_filtered_lemmas, get_lemmas_dictionary, tsne_dim_reduction, umap_dim_reduction)


class ModelOptimizer:
    """
    Class for optimizing topic modeling parameters.

    Parameters
    ----------
    df: pd.DataFrame
        Data frame with text data.
    id_column: str
        Name of the column with document ids.
    section_column: str
        Name of the column with section names.
    column_filter: Dict[str, str]
        Dictionary with column names and values to filter data frame.
    model_type: str, default "lda"
        Type of topic modeling model. Can be "lda", "lsi", "hdp", "rp", "nmf", "d2v".
    words_to_remove: List[str], default []
        List of words to remove from text data.
    topic_numbers_range: Tuple[int, int], default (2, 11)
        Range of topic numbers to try.
    coherence_measure: str, default "c_v"
        Coherence measure to use. Can be "c_v", "c_uci", "c_npmi", "u_mass".
    coherence_num_words: int, default 20
        Number of words used for calculating coherence measure.
    random_state: Optional[int], default None
        Random state for reproducibility.
    name: Optional[str], default None
        Name of model optimizer.
    **kwargs:
        Additional parameters for topic modeling model.
    """
    def __init__(
            self,
            df: pd.DataFrame,
            id_column: str,
            section_column: str,
            column_filter: Dict[str, str],
            model_type: str = "lda",
            words_to_remove: List[str] = [],
            topic_numbers_range: Tuple[int, int] = (2, 11),
            coherence_measure: str = "c_v",
            coherence_num_words: int = 20,
            random_state: Optional[int] = None,
            name: Optional[str] = None,
            **kwargs):
        self.model_type = model_type
        self.id_column = id_column
        self.section_column = section_column
        self.column_filter = column_filter
        self.random_state = random_state
        self.coherence_measure = coherence_measure
        self.coherence_num_words = coherence_num_words
        self.data = df.loc[(df[list(column_filter)] == pd.Series(column_filter)).all(axis=1)]
        self.docs = self.data["text"]
        self.filtered_lemmas = get_filtered_lemmas(self.data, words_to_remove)
        self.lemmas_dictionary = get_lemmas_dictionary(self.filtered_lemmas)
        self.encoded_docs = self.filtered_lemmas.apply(self.lemmas_dictionary.doc2bow)
        self.models = get_models(self.docs, self.encoded_docs, self.filtered_lemmas, self.model_type,
                                 topic_numbers_range, random_state, **kwargs)
        self.cvs = get_coherences(self.models, self.filtered_lemmas, self.lemmas_dictionary, self.coherence_measure,
                                  self.coherence_num_words)
        self.topics_num = get_best_topics_num(self.cvs)
        self.topic_names_dict = {i: i for i in range(self.topics_num)}
        self.name = name

    @property
    def best_model(self) -> Model:
        return self.models[self.topics_num]

    def get_topics_df(self, num_words: int = 10) -> pd.DataFrame:
        """Returns data frame with topic words and their importances."""
        result, is_word = self.best_model.get_topics(num_words=num_words)
        counter = Counter(self.filtered_lemmas.sum())
        id2word_dict = self.lemmas_dictionary
        if not is_word:
            result.iloc[:, 1] = result.iloc[:, 1].map(id2word_dict)
        result["word_count"] = result.iloc[:, 1].map(counter)
        result.columns = ["topic_id", "word", "importance", "word_count"]
        result = result.sort_values(by=["importance"], ascending=False)
        return result

    def get_topic_probs_df(self) -> pd.DataFrame:
        """Returns original data frame with added columns for topic probabilites."""
        res = self.best_model.get_topic_probs(self.encoded_docs)
        modeling_results = pd.concat([self.data.reset_index(drop=True), pd.DataFrame(res)], axis=1)

        return modeling_results

    def get_topic_probs_averaged_over_column(
        self,
        column: Optional[str] = None,
        show_names: bool = False,
    ) -> pd.DataFrame:
        """Returns topic probabilities averaged over given column."""
        if not column:
            column = self.id_column
        modeling_results = self.get_topic_probs_df()
        result = []
        column_vals_added = []
        column_vals = modeling_results[column].unique()
        rows_by_column = modeling_results.groupby(column).count()[0].max()
        for column_val in column_vals:
            df_tmp = modeling_results[modeling_results[column] == column_val]
            if df_tmp.shape[0] != rows_by_column:
                warnings.warn(f"{column} - {column_val} has missing rows!")
                continue
            result.append(df_tmp.iloc[:, -self.topics_num:].values.flatten())
            column_vals_added.append(column_val)
        res = pd.DataFrame(np.vstack(result), index=column_vals_added)
        res.index.name = column
        if show_names:
            if not len(res.columns) % self.topics_num:
                res.columns = len(
                    res.columns) // self.topics_num * [self.topic_names_dict[i] for i in range(self.topics_num)]
        return res

    def get_tsne_mapping(
        self,
        column: Optional[str] = None,
        perplexity: int = 40,
        n_iter: int = 1000,
        init: str = "pca",
        learning_rate: Union[str, float] = "auto",
    ) -> pd.DataFrame:
        """Returns mapping for t-SNE dimension reduction."""
        if not column:
            column = self.id_column
        topics_by_column = self.get_topic_probs_averaged_over_column(column)
        mapping = tsne_dim_reduction(topics_by_column, self.random_state, perplexity, n_iter, init, learning_rate)
        return mapping

    def get_umap_mapping(
        self,
        column: Optional[str] = None,
        n_neighbors: int = 7,
        metric: str = "euclidean",
        min_dist: float = 0.1,
        learning_rate: float = 1,
    ) -> pd.DataFrame:
        """Returns mapping for UMAP dimension reduction."""
        if not column:
            column = self.id_column
        topics_by_column = self.get_topic_probs_averaged_over_column(column)
        mapping = umap_dim_reduction(
            topics_by_column,
            self.random_state,
            n_neighbors,
            metric,
            min_dist,
            learning_rate,
        )
        return mapping

    def save(self, path: str = ""):
        """Saves model (encoded docs, dictionary, and model) to given path."""
        filter_name = "_".join([
            value.replace(":", "").replace(" ", "_").replace(",", "").replace("/", "_").replace("(", "").replace(
                ")", "").replace("&", "").replace("-", "_") for value in self.column_filter.values()
        ])
        self.encoded_docs.to_csv(path + filter_name + "_encoded_docs.csv")
        self.lemmas_dictionary.save(path + filter_name + "_dictionary.dict")
        self.best_model.save(path + filter_name + "_lda_model.model")

    def name_topics_automatically(
        self,
        num_keywords: int = 15,
        gpt_model: str = "text-davinci-003",
        temperature: int = 0.5,
    ):
        """Generate topic names using GPT model."""
        if openai.api_key == None:
            warnings.warn("""
                Topic names not updated: no api key set;
                Key can be set using function set_openai_key(key)
                Organization can be set using function set_openai_organization(organization)
                """)
            return
        topics_keywords = self.get_topics_df(num_keywords)
        exculded = []
        for i in range(self.topics_num):
            keywords = topics_keywords[topics_keywords["topic_id"] == i].word.to_list()
            weights = topics_keywords[topics_keywords["topic_id"] == i].importance.to_list()
            prompt = _generate_prompt(keywords, weights, exculded)
            title = _generate_title(prompt, gpt_model, temperature)
            self.topic_names_dict[i] = title
            exculded.append(title)

    def name_topics_manually(self, topic_names: Union[List[str], Dict[int, str]]):
        """Manually name topics."""
        if isinstance(topic_names, list):
            dict_update = {i: topic_names[i] for i in range(len(topic_names))}
        if isinstance(topic_names, dict):
            dict_update = topic_names
        updated_dict = deepcopy(self.topic_names_dict)
        updated_dict.update(dict_update)
        if updated_dict.keys() == self.topic_names_dict.keys():
            self.topic_names_dict = updated_dict
        else:
            warnings.warn("Topic names not updated: incorrect topic names given")

    def get_topic_names(self):
        """Prints topic names."""
        for topic_id in range(len(self.topic_names_dict)):
            print(f"{topic_id}:", self.topic_names_dict[topic_id])


def get_best_topics_num(cvs: Dict[int, float]) -> int:
    """Returns best number of topics based on coherence values."""
    return max(cvs, key=cvs.get)


def get_models(
        docs: Union[pd.Series, List[List[str]]],
        encoded_docs: Union[pd.Series, List[List[str]]],
        filtered_lemmas: Union[pd.Series, List[List[str]]],
        model_type: str = "lda",
        topic_numbers_range: Tuple[int, int] = (2, 11),
        random_state: Optional[int] = None,
        **kwargs,
) -> Dict[int, Model]:
    """Returns dictionary of models with keys as number of topics."""
    return {
        num_topics: Model(num_topics=num_topics,
                          docs=docs,
                          filtered_lemmas=filtered_lemmas,
                          encoded_docs=encoded_docs,
                          model_type=model_type,
                          random_state=random_state,
                          **kwargs)
        for num_topics in tqdm(range(*topic_numbers_range))
    }


def get_coherences(
    models: Dict[int, Model],
    texts: Union[pd.Series, List[List[str]]],
    dictionary: Dictionary,
    coherence: str = "c_v",
    num_words: int = 40,
) -> Dict[int, float]:
    """Returns dictionary of coherence values with keys as number of topics."""
    return {
        num_topics: CoherenceModel(topics=model.get_topics_list(dictionary=dictionary, num_words=num_words),
                                   texts=texts,
                                   dictionary=dictionary,
                                   coherence=coherence).get_coherence()
        for num_topics, model in tqdm(models.items())
    }


def set_openai_key(key: str):
    """Sets OpenAI api key."""
    openai.api_key = key


def _generate_prompt(keywords: list, weights: list, excluded: list = []) -> str:
    """Generates prompt for GPT-3 model."""
    keywords_weights = [word + ": " + str(weight) for word, weight in zip(keywords, weights)]
    if len(excluded) > 0:
        excluded_str = f". Desctription must be different than: {', '.join(excluded)} "
    else:
        excluded_str = ""
    return ("Describe topic in maximum three words based on given keywords and their importance: " +
            ", ".join(keywords_weights) + excluded_str)


def _generate_title(prompt: str, gpt3_model: str, temperature: int) -> str:
    """Generates title for topic using GPT-3 model."""
    response = openai.Completion.create(model=gpt3_model, prompt=prompt, temperature=temperature)
    return response.choices[0].text.split("\n")[-1].replace('"', '')