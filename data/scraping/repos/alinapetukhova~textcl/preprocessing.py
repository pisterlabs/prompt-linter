"""
This module contains functions for general text preprocessing and filtering.
"""

import pandas as pd
import re
from langdetect import detect_langs
import math
import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import warnings
warnings.filterwarnings("ignore", '.*future version*')


def perplexity_filtering(sentences_df, threshold=1000, sentence_col="sentence"):
    """
    Function used to filter sentences by perplexity

    ---

    **Arguments**\n
    `sentences_df` (DataFrame): DataFrame with sentences and which contains *sentence* column.\n
    `threshold` (int): Perplexity threshold used for filtering. Default value = 1000.\n
    `sentence_col` (String): Name of the sentence column in data frame. Default value = "sentence".

    ---

    **Returns**\n
    `sentences_df` (DataFrame): DataFrame filtered by perplexity.
    """

    # Load pre-trained model (weights)
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    def score(sentence):
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss=model(tensor_input, lm_labels=tensor_input)
        return math.exp(loss.item())

    l = list(sentences_df)
    sentences_df['perplexity'] = sentences_df[sentence_col].apply(lambda x: score(x) if len(re.sub('[^0-9a-zA-Z ]', '', x)) > 0 else -1.0)
    return sentences_df[(sentences_df['perplexity'] <= threshold) & (sentences_df['perplexity'] != - 1.0)][l]


def language_filtering(sentences_df, threshold=0.99, language='en', sentence_col="sentence"):
    """
    Function used to filter sentences by language

    ---

    **Arguments**\n
    `sentences_df` (DataFrame): DataFrame with sentences and which contains *sentence* column. \n
    `threshold` (float): Language score threshold used for filtering. Default value = 0.99. \n
    `language` (String, optional): Language of sentences. Default value = 'en'. \n
    `sentence_col` (String): Name of the sentence column in data frame. Default value = "sentence".

    ---

    **Returns**\n
    `sentences_df` (DataFrame): DataFrame filtered by language.
    """

    def language_score(sentence, language = language):
        lang_score = 0
        try:
            language_detection_result = detect_langs(sentence)
            for result in language_detection_result:
                if result.lang == language:
                    lang_score = result.prob
        except:
            warnings.warn('Problem with detecting language for the sentence')
        return lang_score

    l = list(sentences_df)
    sentences_df['lang_score'] = sentences_df[sentence_col].apply(lambda x: language_score(x, language))
    return sentences_df[sentences_df['lang_score'] > threshold].reset_index(drop=True)[l]


def jaccard_sim_filtering(sentences_df, sentece_col="sentence", threshold=0.8):
    """
    Function used to filter sentences by Jaccard similarity

    ---

    **Arguments**\n
    `sentences_df` (DataFrame): DataFrame with sentences and which contains *sentence* column. \n
    `sentence_col` (String): Name of the sentence column in data frame. Default value = "sentence".\n
    `threshold` (float): Jaccard similarity score threshold used for filtering. Default value = 0.8.

    ---

    **Returns**\n
    `sentences_df` (DataFrame): DataFrame filtered by Jaccard similarity.
    """

    sentence_set_list = sentences_df[sentece_col].str.split(' ').apply(lambda x: set(x)).values
    for i in range(0, len(sentence_set_list), 1):
        for j in range(i + 1, len(sentence_set_list), 1):
            a = sentence_set_list[i]
            b = sentence_set_list[j]
            c = a.intersection(b)
            sim_score = float(len(c)) / (len(a) + len(b) - len(c))
            if sim_score > threshold:
                sentences_df.loc[i, sentece_col] = 'FILTERED'
                break
    return sentences_df[~(sentences_df[sentece_col] == 'FILTERED')].reset_index(drop=True)


def join_sentences_by_label(grouped_sentences_df, label_col="topic_name", sentence_col="sentence"):
    """
    Function used to join sentences into texts. Sentences are grouped by topics

    ---

    **Arguments**\n
    `grouped_sentences_df` (DataFrame): DataFrame with sentences groped by topics and which contains
    *label_col*, *sentence_col* columns. \n
    `label_col` (String): Name of the label column in data frame. Default value = "topic_name".\n
    `sentence_col` (String): Name of the sentence column in data frame. Default value = "sentence".\n

    ---

    **Returns**\n
    `joined_df` (DataFrame): DataFrame with columns *label_column_name*, *joined_sentences*.
    """

    return grouped_sentences_df.groupby([label_col])[sentence_col].apply(' '.join).reset_index()


def split_into_sentences(text_df, text_col="text", sentence_col="sentence"):
    """
    Function used to texts into sentences

    ---

    **Arguments**\n
    `text_df` (DataFrame): DataFrame with search results which contains *topic_name*, *document_id*, *text*.
    `text_col` (String): Name of the text column in data frame. Default value = "text".\n
    `sentence_col` (String): Name of the sentence column in data frame. Default value = "sentence".\n

    **Returns**\n
    `text_df` (DataFrame): DataFrame with the same structure as text_df and with *sentence* columns.
    """
    text_df[sentence_col] = text_df[text_col].str.strip().replace(r'\s*•', '.').str.split(r'(?<=[.!?…]) ')
    return text_df.explode(sentence_col).reset_index(drop = True)

