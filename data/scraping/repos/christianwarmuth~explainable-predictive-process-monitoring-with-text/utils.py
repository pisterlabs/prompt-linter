import os
import math
import pickle
import re
from typing import List
from typing import Tuple
import toml

import en_core_web_sm
import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
# import pyLDAvis
from gensim.models import CoherenceModel
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

nltk.download("stopwords")
nlp = en_core_web_sm.load()
nlp.max_length = 1500000

keyword_map_title = {"Home improvement": ["home", "bedroom", "bathroom", "basement", "kitchen", "floor",
                                          "property", "house", "relocation", "remodel",
                                          "renovation", "apartment"],
                     "Student Loan": ["student", "fee", "university", "tuition", "school", "degree", "class", "grad",
                                      "graduate"],
                     "Consume": ["mustang", "car", "machine", "auto", "purchase", "replacement", "sport", "christmas",
                                 "game", "gift", "bike", "scooter"],
                     "Medical": ["hospital", "cancer", "medical", "doctor", "uninsured",
                                 "medicine", "surgery", "insurance", "drug", "treatment", "dental"],
                     "Vacation": ["vacation", "summer", "winter", "country", "travel", "family", "wedding", "ring",
                                  "swim", "pool", "hotel"],
                     "Consolidation": ["refinance", "debt", "interest", "consolidation", "banks", "rate", "cut",
                                       "payoff", "limit", "reduction", "credit"],
                     }

config = toml.load("../config.toml")


########################################################################################################################
# NLP Data Preparation Utils
########################################################################################################################

signs = ['{', '}', '!', '\"', '§', '$', '%', '&', '(', ')', '=', '?', '\\',
         '*', '+', '\'', '#', ';', ',', '<', '>', '|', '^', '°', '[', ']']

stop_words = stopwords.words('english')


def remove_symbols_wo_semantic(corpus: List[str]) -> List[str]:
    """
    Remove Not Semantically Rich Words

    :param corpus: Corpus of Documents
    :return: Corpus of Preprocessed Documents
    """

    corpus_without_signs = []
    for word_list in corpus:
        word_list_new = []
        for word in word_list:
            word_new = ''
            for letter in str(word):
                if letter not in signs:
                    word_new = word_new + letter
            if word_new != '':
                word_list_new.append(word_new)
        corpus_without_signs.append(word_list_new)
    return corpus_without_signs


def filter_out_numbers(corpus: List[str]) -> List[str]:
    """
    Filter Single Numbers from Corpus

    :param corpus: corpus of Documents
    :return: Corpus of Preprocessed Documents
    """

    corpus_new = []
    for wordlist in corpus:
        word_list_new = [x for x in wordlist if not any(c.isdigit() for c in x)]
        corpus_new.append(word_list_new)
    return corpus_new


def filter_out_empty_strings(corpus: List[str]) -> List[List[str]]:
    """
    Filter Empty Strings After Other Preprocessing Steps

    :param corpus: Corpus of Documents
    :return: Corpus of Preprocessed Documents
    """

    corpus_new = []
    for wordlist in corpus:
        word_list_new = [x.strip() for x in wordlist if x.strip()]
        corpus_new.append(word_list_new)
    return corpus_new


def remove_single_chars_num(corpus: List[str]) -> List[List[str]]:
    """
    Remove Single Chars from Corpus

    :param corpus: Corpus of Documents
    :return: Corpus of Preprocessed Documents
    """
    corpus_new = []

    for document in corpus:
        document_new = []
        for word in document:
            if word.isdigit() == False:
                if len(word) > 1:
                    document_new.append(word)
        corpus_new.append(document_new)

    return corpus_new


def remove_stopwords(corpus: List[str]) -> List[List[str]]:
    """
    Remove Stopwords from Corpus

    :param corpus: Corpus of Documents
    :return: Corpus of Preprocessed Documents
    """
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in corpus]


def lemmatization(texts: List[str]) -> List[str]:
    """
    Lemmatize Words (Grouping of Inflected Forms of Words)

    :param texts: Corpus of Documents
    :return: Corpus of Preprocessed Documents
    """

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc])
    return texts_out


def make_bigrams(corpus: List[str]) -> List[str]:
    """
    Create Bigrams from Corpus
    :param corpus: Corpus of Documents
    :return: Corpus of Bigrams
    """

    bigram = gensim.models.Phrases(corpus, min_count=5, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in corpus]


def make_trigrams(corpus: List[str]) -> List[str]:
    """
    Create Trigrams from Cropus
    :param corpus: Corpus of Documents
    :return: Corpus of Trigrams
    """

    bigram = gensim.models.Phrases(corpus, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[corpus], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[doc] for doc in corpus]


def preprocessing_pipeline(corpus: List[List[str]], bigrams: bool = True):
    """
    Perfrom NLP Preprocessing Pipeline
    :param corpus: Corpus of Documents
    :param bigrams: Bigrams or Trigrams as Representation
    :return: Preprocessed Corpus
    """

    corpus = remove_symbols_wo_semantic(corpus)
    corpus = filter_out_numbers(corpus)
    corpus = filter_out_empty_strings(corpus)
    corpus = remove_stopwords(corpus)
    corpus = lemmatization(corpus)
    corpus = remove_single_chars_num(corpus)
    if bigrams:
        corpus = make_trigrams(corpus)
    else:
        corpus = make_trigrams(corpus)
    return corpus


def clean_desc(desc: str) -> str:
    """
    Remove >Borrower added< Tag and further HTML tags
    :param desc: input document
    :return: document without HTML tags
    """

    if pd.isna(desc):
        return desc
    else:
        desc = re.sub(
            "^\s*Borrower added on \d\d/\d\d/\d\d > |<br>", lambda x: " ", desc
        ).strip()
        return re.sub(
            "<br>", lambda x: " ", desc
        ).strip()


########################################################################################################################
# UI Elements for Prototypical UI
########################################################################################################################

def n_important_words_wordcloud(shap_values, n: int = 200):
    """
    Create Wordcloud for n Feature Importance Values
    :param shap_values: SHAP Feature Importance Values
    :param n: Number of Words
    :return: Matplotlib Figure of Wordlcoud
    """

    feature_importances = get_n_important_features(shap_values, n=n, with_importance=True)
    zip_iterator = zip(feature_importances[0], feature_importances[1])
    a_dictionary = dict(zip_iterator)

    def col(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
        return "rgb({}, {}, {})".format(0, 0, 0)

    wordcloud = WordCloud(width=1600, height=800, background_color="white", repeat=False, prefer_horizontal=1,
                          relative_scaling=.1, color_func=col, max_font_size=150).generate_from_frequencies(
        a_dictionary)
    plt.figure(figsize=(32, 24), dpi=300)
    plt.imshow(wordcloud)
    plt.axis("off")
    return plt


########################################################################################################################
# Rediscovery Rate Calculation
########################################################################################################################

def get_n_important_features(shap_values, n: int = 200, remove_stop_words=True, with_importance=False) -> List[str]:
    """
    Calculate Most Important Features from SHAP Values

    :param shap_values: SHAP Feature Importance Values
    :param n: Number of Words
    :param remove_stop_words: Remove Stopwords from Feature Importance Values
    :param with_importance: Return Importance Values alongside Feature Names
    :return: Most Important Feature Names
    """

    features = shap_values.data
    feature_names = shap_values.feature_names

    cohorts = {"": shap_values}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
    important_features = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=['feature_name', 'mean_abs_shap_importance'])
    len(values)
    important_features.sort_values(
        by=['mean_abs_shap_importance'], ascending=False, inplace=True)

    if remove_stop_words:
        important_features = important_features[
            (~important_features['feature_name'].isin(stop_words)) & (important_features['feature_name'].str.len() > 3)]
    if with_importance:
        if n is not None:
            return important_features["feature_name"].iloc[:n].tolist(), important_features[
                                                                             "mean_abs_shap_importance"].iloc[
                                                                         :n].tolist()
        else:
            return important_features["feature_name"].tolist(), important_features[
                "mean_abs_shap_importance"].tolist()
    if n is not None:
        return important_features["feature_name"].iloc[:n].tolist()
    else:
        return important_features["feature_name"].tolist()


def to_prob(importance: float) -> float:
    """
    Convert Log-Odds to Probability
    :param importance: Log-Odds Input
    :return: Probability Output
    """

    return math.exp(importance) / (1 + math.exp(importance))


def to_prob_list(importance: List[float]) -> List[float]:
    """
    Convert List of Log-Odds Values to List of Probability Values
    :param importance: List of Log-Odds Input
    :return: List of Probability Output
    """

    return [math.exp(importance[0]) / (1 + math.exp(importance[0])),
            math.exp(importance[1]) / (1 + math.exp(importance[1]))]


def calculate_overlap(feature_names, importance, input_words, sim_cutoff=0.25):
    lemmatizer = WordNetLemmatizer()

    important_features = list(map(lambda x: lemmatizer.lemmatize(x), feature_names))
    input_words = list(map(lambda x: lemmatizer.lemmatize(x), input_words))

    vectors = KeyedVectors.load_word2vec_format(config["data"]["word2vec"], binary=True)

    matches = 0
    for word in input_words:
        sim = 0
        match = ""
        for idx, imp_word in enumerate(important_features):
            if round(importance[idx], 1) >= 0.005:

                if (word in vectors.vocab and imp_word in vectors.vocab):
                    if vectors.similarity(word, imp_word) > sim:
                        sim = vectors.similarity(word, imp_word)
                        match = imp_word
        if sim > sim_cutoff:
            # print(word + " " + match)
            matches += 1

    return matches / len(input_words)


########################################################################################################################
# Model Evaluation
########################################################################################################################

def plot_confusion_matrix(confusion_matrix,
                          classes: List[str] = ["Rejected", "Accepted"]) -> plt.Figure:
    """
    Plot Confusion Matrix with Normalized and Absolute Values
    :param y_test: Ground-truth Labels
    :param y_pred: Predicted values
    :param classes: Textual Descriptions of Classes
    :return: Figure
    """

    cm = confusion_matrix
    normalized_values = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    flat_values = ["{0:0.0f}".format(value) for value in cm.flatten()]
    percentages = ["{0:.1%}".format(value) for value in normalized_values.flatten()]

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(flat_values, percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    fig, ax = plt.subplots()
    sns.heatmap(normalized_values, annot=labels, fmt='', cmap='Blues', xticklabels=classes, yticklabels=classes)

    ax.set(ylabel='True label', xlabel='Predicted label', title='Confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.yticks(rotation=0)

    fig.set_size_inches(5, 4.3)
    fig.tight_layout()


def pickl_save(location, obj):
    pickl_location = location
    with open(pickl_location, "wb") as f:
        pickle.dump(obj, f)


########################################################################################################################
# Experiment Evaluation
########################################################################################################################


def calculate_prediction_strat1(logits_1: List[float], class_proba_2: List[float]) -> Tuple[
    List[int], List[int], List[int]]:
    """
    Calculate Combined Model Prediction of Strategy 1
    :param logits_1: Logits Model 1
    :param class_proba_2: Class Probabilities Model 2
    :return: Predictions, Predictions MOdel 1, Predictions Model 2
    """

    logits_bert = logits_1.tolist()
    proba_bert = list(map(to_prob_list, logits_bert))
    bert_class_membership = np.argmax(logits_bert, axis=1)
    xgb_class_membership = np.argmax(class_proba_2, axis=1)
    proba_xgb = class_proba_2.tolist()

    matches = []
    major = []
    for i in range(len(proba_bert)):
        abs_bert = abs(0.5 - proba_bert[i][0]) + abs(0.5 - proba_bert[i][1])
        abs_xgb = abs(0.5 - proba_xgb[i][0]) + abs(0.5 - proba_xgb[i][1])
        if abs_bert > abs_xgb:
            major.append(0)
            matches.append(np.argmax(proba_bert[i]))
        else:
            major.append(1)
            matches.append(np.argmax(proba_xgb[i]))

    return matches, bert_class_membership, xgb_class_membership


def calculate_rediscovery_metric(shap_values, min_importance=0.01) -> float:
    """
    Calculate Rediscovery Score (Cosine Similarity Between Word2vec Vectors)
    :param shap_values: SHAP Feature Importance Values
    :return: Rediscovery Score
    """

    positive = []
    negative = []
    for topic in ["Home improvement", "Consume", "Vacation"]:
        positive.extend(keyword_map_title[topic])

    for topic in ["Student Loan", "Medical", "Consolidation"]:
        negative.extend(keyword_map_title[topic])

    feature_names, importance = get_n_important_features(shap_values, n=None, with_importance=True)

    rediscovery_score = calculate_overlap(feature_names, importance, positive + negative, 0.30)
    return rediscovery_score
