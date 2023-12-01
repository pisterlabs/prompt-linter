from pandas_profiling import ProfileReport
from dotenv import load_dotenv
import os
import pyodbc
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
import spacy
import pandas as pd
import numpy as np
#from rake_nltk import Rake
from bertopic import BERTopic
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
import typing as t
from typing import List
from nltk.corpus import stopwords
import logging
import plotly.graph_objs as go
from bertopic import BERTopic
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px


###############################################################
################# Globale Variablen  ##########################
###############################################################

# plot settings
#color_continuous_scale="darkmint"
#color_discrete_sequence=['#001d13', '#002822', '#00342f', '#00403a', '#004d46', '#005952', '#00675f', '#00746c', '#008279', '#009087', '#009e94', '#14aca2', '#31bab0', '#45c8bd', '#56d6cb', '#66e5d9', '#75f3e7', '#91ffff']
#color_discrete_sequence_mixed=['#001d13', '#91ffff', '#002822', '#75f3e7', '#00342f', '#66e5d9', '#00403a', '#56d6cb', '#004d46', '#45c8bd', '#005952', '#31bab0', '#00675f', '#14aca2', '#00746c', '#009e94', '#008279', '#009087']

#color_discrete_sequence=["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600", "#ff7c43", "#ffdc00", "#00a2ff", "#7fdbff", "#e8c547", "#55b2d2", "#7fcdbb", "#5a5a5a", "#9c9c9c", "#c9c9c9", "#ef476f", "#6b5b95", "#b565a7", "#ffdab9", "#4d4d4d"]
#color_discrete_sequence=["#0B1F26","#3F89A6","#204959","#96C6D9","#D0E9F2","#42323A","#6C8C7D","#8EB3A2","#C5D9BA","#546E75"]
color_discrete_sequence= [ "#0B1F26",  "#3F89A6",  "#204959",  "#96C6D9",  "#D0E9F2",  "#42323A",  "#6C8C7D",  "#8EB3A2",  "#C5D9BA",  "#546E75",  "#F08080",  "#FFA07A",  "#FFDAB9",  "#FFA500",  "#FFD700",  "#DAA520",  "#BDB76B",  "#808000",  "#556B2F",  "#8B4513"]
color_discrete_sequence_mixed= ['#0B1F26', '#8B4513', '#3F89A6', '#556B2F', '#204959', '#808000', '#96C6D9', '#BDB76B', '#D0E9F2', '#DAA520', '#42323A', '#FFD700', '#6C8C7D', '#FFA500', '#8EB3A2', '#FFDAB9', '#C5D9BA', '#F08080', '#546E75', '#FFA07A']
color_discrete_kuzu= ["#49787F","#A96262","#F06969","#C499CA","#EDB183","#B6D6CC","#9D6E94","#4FB286","#87A07C","#74A4BC","#F0C7ED","#9C89B8","#F06969","#ECD9BD"]

template='plotly_white'




################################################################
######################## Functions #############################
################################################################


def sql_azure_connect():
    '''
    Connect to an Azure SQL database using credentials from a .env file.

    Returns:
        cnxn (pyodbc.Connection): A connection object to the Azure SQL database.

    Raises:
        pyodbc.Error: If the connection to the database fails.
    '''

    # Load .env file
    load_dotenv('../config/.env')

    # Import credentials for kuzu Azure DB from .env file
    credentials = {
        'SERVER': os.getenv('SERVER_AZURE', "default"),  
        'DATABASE': os.getenv('DATABASE_AZURE', "default"),
        'USERNAME': os.getenv('USERNAME_AZURE', "default"),
        'PASSWORD': os.getenv('PASSWORD_AZURE', "default"),
        'DRIVER': os.getenv('DRIVER_AZURE', "default") 
    }

    connection_string = f"DRIVER={credentials['DRIVER']};SERVER={credentials['SERVER']};DATABASE={credentials['DATABASE']};UID={credentials['USERNAME']};PWD={credentials['PASSWORD']}"
    cnxn = pyodbc.connect(connection_string)

    # Show available tables
    table_names = [x[2] for x in cnxn.cursor().tables(tableType='TABLE')]
    print("Available tables: ",table_names)

    return cnxn 


###########################################


def filter_dateframe_cols(df,cols:list):
    df = df[[cols]]


###########################################


def add_basic_textfeatures(df, colname: str):
    '''
    Add basic text features to a dataframe column with text.

    Args:
        df (pandas.DataFrame): The input dataframe.
        colname (str): The name of the column with the text.

    Returns:
        pandas.DataFrame: A new dataframe with the added text features.

    Examples:
        >>> df = pd.DataFrame({'text': ['This is a sentence.', 'This is another sentence.']})
        >>> df_with_features = add_basic_textfeatures(df, 'text')
        >>> print(df_with_features)
                    text  text_Character  text_Tokens  text_Types  text_TTR
        0  This is a sentence.              19            4           4    100.00
        1  This is another sentence.      26            4           4    100.00
    '''

    dff = df.copy()

    ## Add character count
    dff[colname + '_' + 'Character'] = df[colname].apply(lambda x: len(x))

    ## Add token count (wordcount)
    dff[colname + '_' + 'Tokens'] = df[colname].apply(lambda x: len(str(x).split()))

    ## Add types count (unique wordcount)
    typecount = df[colname].apply(lambda x: len(set(str(x).split())))
    dff[colname + '_' + 'Types'] = typecount

    ## Add TTR (Type-Token Ratio)
    dff[colname + '_' + 'TTR'] = (typecount / dff[colname + '_' + 'Tokens']) * 100

    return dff


###########################################


def remove_redundant_whitespaces(column):
    '''Removes all additional whitespaces from a list ans returns a new list'''
    
    return [re.sub(r'\s+'," ", x).strip() for x in column]


###########################################

def get_top_n_ngrams(corpus, n=None, ngram_range=(1,1)):
    '''
    Get the top n n-grams from a corpus of text.

    Args:
        corpus (list or array-like): The input corpus of text.
        n (int or None): The number of n-grams to return. If None, return all n-grams.
        ngram_range (tuple): The range of n-grams to consider. Default is (1,1) for unigrams.

    Returns:
        list: A list of tuples, where each tuple contains an n-gram and its frequency in the corpus, sorted by frequency in descending order.

    Examples:
        >>> corpus = ['This is a sentence.', 'This is another sentence.']
        >>> top_ngrams = get_top_n_ngrams(corpus, n=2, ngram_range=(1,2))
        >>> print(top_ngrams)
        [('this', 2), ('is', 2)]
    '''

    vec = CountVectorizer(ngram_range=ngram_range)
    # check if corpus is a list of lists and flatten it if so
    if isinstance(corpus[0], list):
        flat_corpus = [item for sublist in corpus for item in sublist]
    else:
        flat_corpus = corpus
    vec.fit(flat_corpus)
    bag_of_words = vec.transform(flat_corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


################## Add addition date dimensions to datarframe ######################### 

def add_date_columns(df, date_col):
    """
    Add additional date columns to a DataFrame based on a specified date column.

    Args:
        df (pandas.DataFrame): The DataFrame to which the new date columns will be added.
        date_col (str): The name of the column containing the date.

    Returns:
        pandas.DataFrame: A new DataFrame with the additional date columns.

    Example:
        >>> df = pd.DataFrame({'date': ['2022-01-01', '2022-02-01', '2022-03-01']})
        >>> df['date'] = pd.to_datetime(df['date'])
        >>> df_with_datecols = add_date_columns(df, 'date')
        >>> print(df_with_datecols.head())
                date  year  month  quarter  yearmonth yearquarter  season yearseason
        0 2022-01-01  2022      1        1 2022-01-01      2022Q1  winter   2022-winter
        1 2022-02-01  2022      2        1 2022-02-01      2022Q1  winter   2022-winter
        2 2022-03-01  2022      3        1 2022-03-01      2022Q1  spring   2022-spring
    """

    df.insert(loc=2, column='year', value=df[date_col].dt.year) #create additional year col for viz
    df.insert(loc=3, column='month', value=df[date_col].dt.month) #create additional month col for viz
    df.insert(loc=4, column='quarter', value=df[date_col].dt.quarter) #create additional quarter col for viz
    df.insert(loc=5, column='yearmonth', value=pd.to_datetime(df[['year', 'month']].assign(DAY=1))) #create additional yearmonth col for viz
    df.insert(loc=6, column='yearquarter', value=df['year'].astype(str) + 'Q' + df['quarter'].astype(str)) #create additional yearquarter col for viz
    df.insert(loc=7, column='season', value=df['month'].apply(lambda x: 'spring' if x in [3, 4, 5] else ('summer' if x in [6, 7, 8] else ('autumn' if x in [9, 10, 11] else 'winter'))))
    df.insert(loc=8, column='yearseason', value=df['year'].astype(str) + '-' + df['season']) # create additional yearseason column for viz
    return df


#######################################
################ TFIDF ################

################ General Group Function  ################


def find_trending_keywords(dataframe, filter_column, text_column, ngram_range=(1, 1), n=10, min_df=100, max_df=0.2):
    """
    Find the top n trending keywords for each value in a specified column of a Pandas DataFrame.

    Args:
    - dataframe (pd.DataFrame): The DataFrame containing the data to analyze.
    - filter_column (str): The name of the column in the DataFrame to group the data by.
    - text_column (str): The name of the column in the DataFrame containing the text to analyze.
    - ngram_range (Tuple[int, int]): The range of n-grams to consider when extracting features from the text data.
    - n (int): The number of top keywords to return for each group.
    - min_df (int): The minimum number of documents that a term must appear in to be considered in the analysis.
    - max_df (float): The maximum proportion of documents that a term can appear in to be considered in the analysis.

    Returns:
    - trending_keywords (Dict[str, List[Tuple[str, float, int, float]]]): A dictionary where each key is a unique value from the specified column and each value is a list of tuples containing the top n keywords for that value, along with their TF-IDF score, count in the group, and normalized count.
    """

    # convert values in filter column to categorical values
    dataframe[filter_column] = dataframe[filter_column].astype('category')

    # add "unknown" category to filter_column categories, if not already present
    if "unknown" not in dataframe[filter_column].cat.categories:
        dataframe[filter_column] = dataframe[filter_column].cat.add_categories("unknown")

    # replace NaN values in filter_column with "unknown"
    dataframe[filter_column].fillna("unknown", inplace=True)

    # Create an empty dictionary to store the top keywords and their counts for each value in filter_column
    trending_keywords = {}

    # Get all values in filter_column
    values = dataframe[filter_column].unique()

    # Convert the tokenized text column to a list of space-separated strings
    text_data = [' '.join(words) for words in dataframe[text_column]]

    # Create a TfidfVectorizer object with the specified n-gram range and min_df parameter
    tfidf_vect = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)

    # Fit and transform the tokenized text column for the whole corpus
    tfidf = tfidf_vect.fit_transform(text_data)

    # Loop over the values
    for value in values:
        # Filter the dataframe for the given value in filter_column
        filter_data = dataframe[dataframe[filter_column] == value]

        # Convert the tokenized text column to a list of space-separated strings
        text_data_filter = [' '.join(words) for words in filter_data[text_column]]

        # Transform the tokenized text column for the given value using the fitted TfidfVectorizer
        tfidf_filter = tfidf_vect.transform(text_data_filter)

        # Compute the sum of TF-IDF scores for each term in the given value
        tfidf_filter = tfidf_filter.sum(axis=0)

        # Create a list of tuples with the term and its TF-IDF score for the group
        keywords = [(term, tfidf_filter[0, index]) for term, index in tfidf_vect.vocabulary_.items()]

        # Filter out terms that have zero TF-IDF scores
        keywords = [kw for kw in keywords if kw[1] > 0]

        # Sort the keywords based on their TF-IDF scores
        keywords.sort(key=lambda x: x[1], reverse=True)

        # Count the occurrence of each keyword in the group
        group_text_data = ' '.join(text_data_filter)
        group_word_count = Counter(group_text_data.split())

        # Create a list of tuples with the term, its TF-IDF score, and count in the group
        keywords_with_count = [(kw[0], kw[1], group_word_count[kw[0]], group_word_count[kw[0]]/len(group_word_count)) for kw in keywords]

        # Store the top n keywords for the given value in the dictionary
        trending_keywords[value] = keywords_with_count[:n]

    # Return the dictionary of top keywords and their counts for each value in filter_column
    return trending_keywords



################ Specific Group Function  ################

def find_trending_keywords_diff_normaized(dataframe, filter_column, text_column, ngram_range=(1, 1), n=10, min_df=100, max_df=0.2):
    """
    Given a Pandas dataframe `dataframe`, a categorical column name `filter_column`, and a text column name `text_column`,
    this function returns a dictionary of top `n` trending keywords with their TF-IDF score differences normalized by group size
    for each unique value in `filter_column`. The keyword extraction is based on the difference in TF-IDF scores between
    the given value and the average of the other values in `filter_column`. Only keywords with positive score differences are
    included in the results. The TF-IDF score differences are normalized by the total count of all words in the group. The
    `ngram_range`, `min_df`, and `max_df` parameters control the behavior of the TfidfVectorizer object used for tokenization
    and weighting of the text data.

    Args:
    - dataframe (Pandas dataframe): The dataframe containing the data to analyze.
    - filter_column (str): The name of the categorical column to group the data by.
    - text_column (str): The name of the text column to extract keywords from.
    - ngram_range (tuple, default=(1,1)): The range of n-grams to extract from the text data.
    - n (int, default=10): The number of top keywords to extract for each group.
    - min_df (int, default=100): The minimum frequency threshold for words to be included in the vocabulary.
    - max_df (float, default=0.2): The maximum document frequency threshold for words to be included in the vocabulary.

    Returns:
    - trending_keywords (dict): A dictionary with the unique values in `filter_column` as keys and a list of top `n`
    trending keywords with their TF-IDF score differences normalized by group size as values. The keyword list for each
    value is sorted by descending order of the TF-IDF score difference.
    """

    # convert values in filter column to categorical values
    dataframe[filter_column] = dataframe[filter_column].astype('category')

    # add "unknown" category to filter_column categories, if not already present
    if "unknown" not in dataframe[filter_column].cat.categories:
        dataframe[filter_column] = dataframe[filter_column].cat.add_categories("unknown")

    # replace NaN values in filter_column with "unknown"
    dataframe[filter_column].fillna("unknown", inplace=True)

    # create an empty dictionary to store the top keywords for each value in filter_column
    trending_keywords = {}

    # get all values in filter_column
    values = dataframe[filter_column].unique()

    # convert the tokenized text column to a list of space-separated strings
    text_data = [' '.join(words) for words in dataframe[text_column]]

    # create a TfidfVectorizer object with the specified n-gram range and min_df parameter
    tfidf_vect = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)

    # fit and transform the tokenized text column for the whole corpus
    tfidf = tfidf_vect.fit_transform(text_data)

    # loop over the values
    for value in values:
        # filter the dataframe for the given value in filter_column
        filter_data = dataframe[dataframe[filter_column] == value]

        # convert the tokenized text column to a list of space-separated strings
        text_data_filter = [' '.join(words) for words in filter_data[text_column]]

        # transform the tokenized text column for the given value using the fitted TfidfVectorizer
        tfidf_filter = tfidf_vect.transform(text_data_filter)

        # compute the sum of TF-IDF scores for each term in the given value
        tfidf_filter = tfidf_filter.sum(axis=0)

        # normalize the TF-IDF scores by the total count of all words in the group
        group_word_count = Counter(' '.join(text_data_filter).split())
        total_count = sum(group_word_count.values())
        tfidf_filter = tfidf_filter / total_count

        # Compute the sum of TF-IDF scores for each term in the other values
        tfidf_other_sum = 0
        for other_value in values:
            if other_value != value:
                # Filter the dataframe for the other value in filter_column
                other_data = dataframe[dataframe[filter_column] == other_value]

                # Convert the tokenized text column to a list of space-separated strings
                text_data_other = [' '.join(words) for words in other_data[text_column]]

                # Transform the tokenized text column for the other value using the fitted TfidfVectorizer
                tfidf_other = tfidf_vect.transform(text_data_other)

                # Compute the sum of TF-IDF scores for each term in the other value
                tfidf_other = tfidf_other.sum(axis=0)

                # normalize the TF-IDF scores by the total count
                total_count = tfidf_other.sum()
                tfidf_other = tfidf_other / total_count

                # Add the normalized TF-IDF scores to the running sum
                tfidf_other_sum += tfidf_other

        # Compute the average of the other values' TF-IDF scores for each term
        tfidf_other_avg = tfidf_other_sum / (len(values) - 1)

        # Compute the difference in TF-IDF scores between the given value and the average of the other values
        tfidf_diff = tfidf_filter - tfidf_other_avg

        # Create a list of tuples with the term and its TF-IDF score difference
        keywords = [(term, tfidf_diff[0, index]) for term, index in tfidf_vect.vocabulary_.items()]

        # Filter out terms that have negative or zero TF-IDF score differences
        #keywords = [kw for kw in keywords if kw[1] > 0]

        # Sort the keywords based on their TF-IDF score difference
        keywords.sort(key=lambda x: x[1], reverse=True)

        # Count the occurrence of each keyword in the group
        group_text_data = ' '.join(text_data_filter)
        group_word_count = Counter(group_text_data.split())

        # Compute the total count of all words in the group
        total_count = sum(group_word_count.values())

        # Create a list of tuples with the term, its TF-IDF score difference, count in the group, and relative count
        keywords_with_count_rel = [(kw[0], kw[1], group_word_count[kw[0]], group_word_count[kw[0]] / total_count) for kw in keywords]

        # Store the top n keywords for the given value in the dictionary with relative count
        trending_keywords[value] = keywords_with_count_rel[:n]

    # Return the dictionary of top keywords for each value in filter_column
    return trending_keywords

############### Transform TF-IDF results for output ###########

def trending_keywords_to_dataframe(trending_keywords):
    """
    Converts a dictionary of trending keywords to a Pandas DataFrame.

    Parameters:
    trending_keywords (dict): A dictionary with group names as keys and lists of
                              (keyword, TF-IDF score, count, relative count) tuples
                              as values.

    Returns:
    Pandas DataFrame: A DataFrame with columns 'Group', 'Keyword', 'Scores',
                      'Total_Group_Count', and 'Relative_Group_Count'. Each row
                      corresponds to a single keyword for a single group, with
                      the values of the respective columns populated by the
                      corresponding elements of the input dictionary.
    """

    rows = []
    for group, keywords in trending_keywords.items():
        for keyword, tfidf_score, count ,relativecount in keywords:
            row = {'Group': group, 'Keyword': keyword, 'Scores': tfidf_score, 'Total_Group_Count': count,'Relative_Group_Count':relativecount}
            rows.append(row)
    return pd.DataFrame(rows)


#####################################################


def convert_resulttable_to_wide_format(df):
    """Converts a result table DataFrame from long to wide format.

    Args:
        df (pandas.DataFrame): A DataFrame with columns 'Group', 'Keyword', and 'Keyword Index'.

    Returns:
        pandas.DataFrame: A DataFrame in wide format with columns for each keyword in each group, and rows for each group.

    The input DataFrame must have the following columns:
    - 'Group': A categorical variable indicating the group to which the keyword belongs.
    - 'Keyword': The keyword itself.
    - 'Keyword Index': A unique identifier for each keyword within each group, in the format 'Keyword <index>'.

    This function pivots the input DataFrame to a wide format, with columns for each keyword in each group and rows for each group.
    The columns are sorted in the same order as the input DataFrame, and the columns are renamed to 'Keyword <index>'.
    The resulting DataFrame is sorted by the order of the groups in the input DataFrame.

    Example:
        >>> df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B'], 'Keyword': ['foo', 'bar', 'baz', 'qux'],
        ...                    'Keyword Index': ['Keyword 1', 'Keyword 2', 'Keyword 1', 'Keyword 2']})
        >>> convert_resulttable_to_wide_format(df)
           Group Keyword 1 Keyword 2
        0     A       foo       bar
        1     B       baz       qux
    """

    # Add a column to indicate the keyword index
    df['Keyword Index'] = 'Keyword ' + (df.groupby('Group').cumcount() + 1).astype(str)

    # Get the order of the groups in the input DataFrame
    group_order = df['Group'].unique()

    # Pivot the DataFrame to a wide format without sorting the columns
    wide_df = df.pivot(index='Group', columns='Keyword Index', values='Keyword')

    # Sort the pivoted DataFrame using the order of the groups in the input DataFrame
    wide_df = wide_df.loc[group_order]

    # Get the order of the columns in the input DataFrame
    column_order = df['Keyword Index'].unique()
    column_order.sort()
    wide_df = wide_df[column_order]  # Use the same order of columns as the input DataFrame

    # Rename the columns
    wide_df.columns = [f'Keyword {i}' for i in range(1, wide_df.shape[1] + 1)]

    # Reset the index
    wide_df = wide_df.reset_index()

    return wide_df

#####################################################


def create_export_table(df, filename=None):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns)),
        cells=dict(values=[df[col] for col in df.columns]))
    ])

    if filename is not None:
        fig.write_html(filename)

    fig.show()

#####################################################

def export_table_to_xlsx(df, filename=None):
    if filename is not None:
        df.to_excel(filename, index=False)
    else:
        # Display the table if filename is not provided
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns)),
            cells=dict(values=[df[col] for col in df.columns]))
        ])
        fig.show()


#####################################
######## Text Preprocessing  ########

def preprocess_text(df, text_column, custom_stopwords=None):
    """
    Preprocesses text in a DataFrame column by performing the following steps:
    - Lowercases the text
    - Removes stop words and custom stopwords (if provided)
    - Removes numbers and special characters
    - Tokenizes the text using the German language model provided by spaCy
    - Lemmatizes the text using the spaCy language model
    - Separates the text into nouns, adjectives, verbs, and a combination of nouns, adjectives, and verbs
    
    Args:
    - df (pandas.DataFrame): DataFrame containing the text to be preprocessed
    - text_column (str): Name of the column containing the text to be preprocessed
    - custom_stopwords (list): List of custom stopwords to be removed from the text. Default is None.
    
    Returns:
    - df (pandas.DataFrame): DataFrame with the preprocessed text and additional columns for tokenized text,
                             lemmatized text, nouns, adjectives, verbs, and a combination of nouns, adjectives,
                             and verbs.
    """

    nlp = spacy.load("de_core_news_lg")
    words_to_remove = set(STOP_WORDS) | set(custom_stopwords) if custom_stopwords else set(STOP_WORDS)

    # Lowercase the text, remove stop words and custom stopwords, and remove numbers and special characters
    text_preprocessed = df[text_column].str.lower().apply(
        lambda x: " ".join([re.sub(r'[^\w\s]', '', word) for word in re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1 \2', x).split() if word not in words_to_remove and not re.search(r'\d', word)])
    )

    tokenized = []
    nouns = []
    adjectives = []
    verbs = []
    nouns_adjectives_and_verbs = []

    for text in text_preprocessed:
        doc = nlp(text)
        if not doc:
            tokenized.append([])
            nouns.append([])
            adjectives.append([])
            verbs.append([])
            nouns_adjectives_and_verbs.append([])
            continue

        tokenized_text = []
        nouns_text = []
        adjectives_text = []
        verbs_text = []
        nouns_adjectives_and_verbs_text = []

        for token in doc:
            if not token.text:
                continue
            token_text = token.text.lower()
            if token_text not in words_to_remove:
                tokenized_text.append(token_text)
                if token.pos_ == "NOUN":
                    nouns_text.append(token_text)
                    nouns_adjectives_and_verbs_text.append(token_text)
                if token.pos_ == "ADJ":
                    adjectives_text.append(token_text)
                    nouns_adjectives_and_verbs_text.append(token_text)
                if token.pos_ == "VERB":
                    verbs_text.append(token_text)
                    nouns_adjectives_and_verbs_text.append(token_text)

        tokenized.append(tokenized_text)
        nouns.append(nouns_text)
        adjectives.append(adjectives_text)
        verbs.append(verbs_text)
        nouns_adjectives_and_verbs.append(nouns_adjectives_and_verbs_text)

    df["text_preprocessed"] = text_preprocessed
    df["text_preprocessed_tokenized"] = tokenized
    df["lemmatized"] = None
    df["nouns"] = nouns
    df["adjectives"] = adjectives
    df["verbs"] = verbs
    df["nouns_adjectives_and_verbs"] = nouns_adjectives_and_verbs

    return df


########### Helper Functions ##########

def split_dataframe(df, datetime_col, train_frac, test_frac, val_frac):
    # sort the dataframe by the selected datetime column
    df = df.sort_values(datetime_col)

    # calculate the number of samples for each subset
    n = len(df)
    train_size = round(n * train_frac)
    test_size = round(n * test_frac)
    val_size = round(n * val_frac)

    # split the dataframe into train and test+val subsets
    train_df, test_val_df = train_test_split(df, test_size=(test_size + val_size), random_state=22, stratify=df[datetime_col])

    # split the test+val dataframe into test and val subsets
    test_df, val_df = train_test_split(test_val_df, test_size=val_size, random_state=22, stratify=test_val_df[datetime_col])

    return train_df, test_df, val_df

#############################################


def join_list_of_list(list_of_list):
    """
    This function takes in a list of lists and returns a list of strings where each string is made by joining the elements of the corresponding list.
        
    Parameters:
        - list_of_list(List[List[Any]]): List of lists whose elements to be joined
            
    Returns:
         List[str]: List of strings where each string is made by joining the elements of the corresponding list.
    """
    return [' '.join(map(str,l)) for l in list_of_list]

#########################################################

def reduce_dataframe(df, group_column, filter_value):
    """
    Reduces a Pandas dataframe based on a specific column and value.
    
    Parameters:
    df (Pandas dataframe): The dataframe to reduce.
    group_column (str): The name of the column to group the dataframe by.
    filter_value: The value to filter the dataframe on.
    
    Returns:
    A reduced Pandas dataframe.
    """
    
    # Group the dataframe by the specified column
    grouped = df.groupby(group_column)
    
    # Filter the groups based on the filter value
    filtered_groups = {group: data for group, data in grouped if filter_value in data[group_column].values}
    
    # Combine the filtered groups into a new dataframe
    reduced_df = pd.concat(filtered_groups.values())
    
    return reduced_df

#######################################################

def check_column_values(df, col1, col2):
    # Check if either of the two columns contains a non-null value
    result = (df[col1].notnull() | df[col2].notnull()).tolist()
    return result


##############################################
############### Pandas Profiling  ############
##############################################


def generate_profiling_report(data_file="DataText", folder_path="data/", report_title=None, report_file="html/ProfilingDataText.html", lazy=False, dark_mode=False, minimal=True):
    """
    Generates a pandas profiling report for the given data file.
    
    Parameters:
    - data_file (str): The name of the data file to be used for generating the report. Default is "DataText".
    - folder_path (str): The path of the folder where the data file is located. Default is "data/".
    - report_title (str): The title to be used for the report. Default is None.
    - report_file (str): The filepath and name of the report file. Default is "html/ProfilingDataText.html"
    - lazy (bool): Whether to load the data in a lazy or non-lazy way. Default is False
    - dark_mode (bool): Whether to use the dark mode or not. Default is False
    - minimal (bool): Whether to produce a minimal report or not. Default is True
    
    Returns:
    None
    """
    # import data
    df = pd.read_feather(folder_path + data_file)
    if report_title is None:
        report_title = data_file
    # Pandas Profiling TextData
    profile = ProfileReport(
        df,
        title=report_title,
        lazy=lazy,
        dark_mode=dark_mode,
        minimal=minimal,
    )
    profile.to_file(report_file)

############################################################
############## Train Topic Model with BERTopic #############
############################################################


def fit_berttopic(target_dir: str, docs: list, embedding_model=None, min_topic_size: int = 50, stop_words=None) -> None:
    """
    Train and save a BERTopic model.

    Args:
        target_dir (str): Directory to save the trained model.
        docs (list): List of documents to train the model on.
        embedding_model (str or object): Name of the embedding model or an object representing the model.
        min_topic_size (int): Minimum size of a topic (HDBSCAN clusters).
        stop_words (list): List of stopwords to use for keyword extraction.
    """

    if not isinstance(docs, list):
        raise ValueError("docs parameter must be a list")

    if not os.path.exists(target_dir):
        try:
            logging.info(f"Fitting BERTopic model with {embedding_model}...")
            german_stop_words = stop_words or stopwords.words("german")
            vectorizer = CountVectorizer(stop_words=german_stop_words)
            model = BERTopic(
                language="german",
                vectorizer_model=vectorizer,
                embedding_model=embedding_model,
                min_topic_size=min_topic_size,
                #nr_topics="auto"
                #umap_model=PCA(n_components=5) #to use PCA as dim reduction
                )
            topics, probs = model.fit_transform(docs)
            #model.fit_transform(docs)
            new_topics = model.reduce_outliers(docs, topics) # Reduce outliers
            model.update_topics(docs, topics=new_topics,vectorizer_model=vectorizer) # update Model

            model.save(target_dir)
            logging.info(f"Model saved to {target_dir}")
        except Exception as e:
            logging.error(f"Error while fitting BERTopic model: {str(e)}")
            raise
    else:
        logging.info(f"Model already trained at {target_dir}")



def fit_berttopic_if_not_exists(target_dir: str, docs: list, embedding_model=None, min_topic_size: int = 50, stop_words=None) -> None:
    """
    Wrapper function for fit_berttopic to avoid retraining a model that has already been trained.

    Args:
        target_dir (str): Directory to save the trained model.
        docs (list): List of documents to train the model on.
        embedding_model (str or object): Name of the embedding model or an object representing the model.
        min_topic_size (int): Minimum size of a topic (HDBSCAN clusters).
        stop_words (list): List of stopwords to use for keyword extraction.
    """

    if not isinstance(docs, list):
        raise ValueError("docs parameter must be a list")

    if os.path.exists(target_dir):
        logging.info(f"Model already trained at {target_dir}")
        return

    fit_berttopic(target_dir, docs, embedding_model, min_topic_size, stop_words)


################## Extract top 10 topic keywords from topic model ##############
def topic_model_top_10_keywords_export(model, modelname, directory_path):
    df_topic_keywords = get_topic_keywords_df(model)
    df_topic_freq = model.get_topic_freq()
    df_topics = pd.merge(df_topic_keywords, df_topic_freq, left_on='topic_id', right_on='Topic', how='left')
    total_count = df_topics['Count'].sum()
    df_topics['Count %'] = ((df_topics['Count'] / total_count) * 100).round(1)
    # reorder columns to place the new column as the second column
    df_topics = df_topics.reindex(columns=['Topic', 'Count', 'Count %', 'keyword 1', 'keyword 2', 'keyword 3', 'keyword 4', 'keyword 5', 'keyword 6', 'keyword 7', 'keyword 8', 'keyword 9', 'keyword 10'])
    file_path = directory_path + "/topic_keywords_"+ modelname+".xlsx"
    df_topics.to_excel(file_path, index=False)

    return df_topics



# def topic_model_top_10_keywords_export(model, modelname, directory_path):
#     df_topic_keywords = get_topic_keywords_df(model)
#     df_topic_freq = model.get_topic_freq()
#     df_topics = pd.merge(df_topic_keywords, df_topic_freq, left_on='topic_id', right_on='Topic', how='left')
#     # reorder columns to place the new column as the second column
#     df_topics = df_topics.reindex(columns=['Topic', 'Count', 'keyword 1', 'keyword 2', 'keyword 3', 'keyword 4', 'keyword 5', 'keyword 6', 'keyword 7', 'keyword 8', 'keyword 9', 'keyword 10'])
#     file_path = directory_path + "/topic_keywords_"+ modelname+".xlsx"
#     df_topics.to_excel(file_path, index=False)

#     return df_topics

################

def get_topic_keywords_df(topic_model):
    """
    Returns a DataFrame with the topics and their keywords.
    
    Parameters:
    -----------
    topic_model: object
        A trained topic model object.
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the topics and their keywords.
    """
    
    # get the topics and their keywords
    topics = topic_model.get_topics()

    # create an empty DataFrame to store the topics and their keywords
    df = pd.DataFrame(columns=['topic_id', 'keyword 1', 'keyword 2', 'keyword 3', 'keyword 4', 'keyword 5', 'keyword 6', 'keyword 7', 'keyword 8', 'keyword 9', 'keyword 10'])

    # loop through each topic and its keywords and add them to the DataFrame
    for i, (topic_id, topic) in enumerate(topics.items()):
        keywords = [word for word, _ in topic]
        df.loc[i] = [topic_id] + keywords + ['']*(10-len(keywords))

    # set the topic_id column as the index
    # df.set_index('topic_id', inplace=True)
    # df.reset_index()
    
    return df

###################################################
############## Topic Model Evaluation #############

def compute_coherence_scores(documents: np.ndarray, bert_models: List[str], coherence_method: str = "u_mass", path: str = "") -> pd.Series:
    cleaned_documents = [doc.replace("\n", " ") for doc in documents]
    cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
    cleaned_documents = [doc if doc != "" else "emptydoc" for doc in cleaned_documents]

    def compute_coherence_score(topics: List[str], topic_words: List[List[str]], coherence_method: str = "u_mass") -> float:
        # Processing taken from BERT model but should be agnostic
        vectorizer = CountVectorizer()

        # Preprocess Documents
        documents = pd.DataFrame({"Document": cleaned_documents, "ID": range(len(cleaned_documents)), "Topic": topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = documents_per_topic.Document.values

        # Vectorizer
        vectorizer.fit_transform(cleaned_docs)
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names_out()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]

        # Evaluate
        coherence_model = CoherenceModel(topics=topic_words, texts=tokens, corpus=corpus, dictionary=dictionary, coherence=coherence_method)
        coherence = coherence_model.get_coherence()
        return coherence

    scores = dict()

    for model_name in bert_models:
        try:
            model = BERTopic.load(path + model_name)

            topics = model.get_document_info(cleaned_documents)["Topic"]
            topic_words = [[words for words, _ in model.get_topic(topic)] for topic in range(len(set(topics)) - 1)]

            coherence = compute_coherence_score(topics=topics, topic_words=topic_words, coherence_method=coherence_method)
            print(f"BERT Model {model_name}: {coherence}")
            scores[model_name] = coherence
        except Exception as e:
            print(f"Failed to evaluate model {model_name}: {str(e)}")

    scores_series = pd.Series(scores)

    return scores_series



def get_topic_ratios(df, timeframe_col, name_col, topic_col):
    """
    Compute the ratio of counts for each combination of CustomName and Topic, 
    aggregated by quarter, relative to the total count for the quarter.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input data containing the columns 'yearquarter', 'CustomName', 'Topic'.
    timeframe_col : str
        The name of the column containing the timeframe information like quarteryear.
    name_col : str
        The name of the column containing the CustomName information.
    topic_col : str
        The name of the column containing the Topic information.
    
    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with the columns 'yearquarter', 'CustomName', 'Topic', 'count_x', 'count_y', 'Topic_Ratio'.
    """
    timeframe_col = timeframe_col

    # Get totals for each quarter
    df_counts_quarter = pd.DataFrame(df.groupby([timeframe_col]).size().reset_index(name='count_y'))
    
    # Aggregate counts by quarter, CustomName, and Topic
    df_topic_quarter = pd.DataFrame(df.groupby([timeframe_col, name_col, topic_col]).size().reset_index(name='count_x'))
    
    # Merge DataFrames
    df_topic_quarter = df_topic_quarter.merge(df_counts_quarter, on=timeframe_col, how='left')
    
    # Compute Topic_Ratio
    df_topic_quarter['Topic_Ratio'] = (df_topic_quarter['count_x'] / df_topic_quarter['count_y'])
    
    return df_topic_quarter[[timeframe_col, 'CustomName', 'Topic', 'count_x', 'count_y', 'Topic_Ratio']]


import pandas as pd

def compute_categorical_counts(df, categorical_col, name_col):
    """
    Compute the counts and relative counts for each combination of CustomName and Topic, 
    aggregated by the categorical column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input data containing the columns for the categorical_col, name_col, and topic_col.
    categorical_col : str
        The name of the column containing the categorical information (e.g. year, quarter, month, etc.).
    name_col : str
        The name of the column containing the CustomName information.
    topic_col : str
        The name of the column containing the Topic information.
    
    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with the columns 'categorical_col', 'CustomName', 'Topic', 'count', and 'relative_count'.
    """
    # Aggregate counts by categorical value, CustomName, and Topic
    df_topic_categorical = pd.DataFrame(df.groupby([categorical_col, name_col]).size().reset_index(name='count'))
    
    # Compute total count for each categorical value
    df_total_count = pd.DataFrame(df.groupby([categorical_col]).size().reset_index(name='total_count'))
    
    # Merge total count into topic counts DataFrame
    df_topic_categorical = df_topic_categorical.merge(df_total_count, on=categorical_col)
    
    # Compute relative count
    df_topic_categorical['relative_count'] = df_topic_categorical['count'] / df_topic_categorical['total_count']
    
    return df_topic_categorical[[categorical_col, name_col, 'count', 'relative_count']]








# def create_grouped_barchart(df, x_col, y_col, color_col, color_discrete_sequence, ignore_group=None, title='', xaxis_title='', yaxis_title='', legend_title='', template=''):
#     if ignore_group:
#         df = df[df[color_col] != ignore_group]
#     fig = px.bar(df,
#                  x=x_col,
#                  y=y_col,
#                  color=color_col,
#                  hover_data=[color_col, y_col,x_col],
#                  color_discrete_sequence=color_discrete_sequence,
#                  template=template,
#                  barmode='group')
    
#     fig.update_layout(
#         width=900, 
#         height=600,
#         title=title,
#         yaxis_title=yaxis_title,
#         xaxis_title=xaxis_title,
#         legend_title=legend_title,
#     )
    
#     fig.update_xaxes(showgrid=False, tickmode='linear', tickangle=0, tickfont=dict(size=12), tickwidth=1)
#     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
#     # wrap long x-axis labels on two lines and rotate by 270 degrees
#     fig.update_layout(
#         xaxis=dict(
#             tickmode='array',
#             tickvals=list(range(len(df[x_col]))),
#             ticktext=[x.replace(' ', '<br>') if len(x) > 40 else x for x in df[x_col]],
#             automargin=True,
#             tickangle=270,
#             tickfont=dict(size=12),
#         ),
#         legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='left', x=0.5)
#     )
    
#     fig.show()

def create_grouped_radar(df, x_col, y_col, color_col, color_discrete_sequence, ignore_group=None, title='', xaxis_title='', yaxis_title='', legend_title='', template=''):
    fig = go.Figure()
    for color in df[color_col].unique():
        if color != ignore_group:
            fig.add_trace(go.Scatterpolar(
                r=df[df[color_col] == color][y_col].values.tolist(),
                theta=df[df[color_col] == color][x_col].values.tolist(),
                fill='none',
                name=color,
                line=dict(color=color_discrete_sequence[df[color_col].unique().tolist().index(color)]),
                showlegend=True,
                marker=dict(size=4)
            ))
    fig.update_layout(
        width=900, 
        height=650,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, df[y_col].max()],
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            angularaxis=dict(
                visible=True,
                tickmode='linear',
                tickfont=dict(size=10),
                showticklabels=True,
                gridcolor='rgba(0,0,0,0.1)'
            )
        ),
        showlegend=True,
        title=title,
        legend_title=legend_title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template=template
    )

    # fig.update_layout(legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5))

    fig.show()






