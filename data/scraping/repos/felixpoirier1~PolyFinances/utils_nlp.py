
import re
import time
    # data analysis
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm_notebook as tqdm
    # natural language processing - NLTK
import spacy
    # natural language processing - Gensim and LDA
import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel
    # natural language processing - TextBlob (Sentiment)
from textblob import TextBlob
    # data visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns




def remove_references(text):
    """
    Removes references at the end of
    speeches, if any. Helper function
    to assist in data cleaning.

    Parameters:
    text (string): speech text

    Returns:
    string: cleaned speech text sans
          references
    """
    references_loc = text.find('\nReferences\n')
    if references_loc != -1:
        text = text[:references_loc]
    return_to_text_loc = text.find('[Rr]eturn to text\n')
    if return_to_text_loc != -1:
        text = text[:return_to_text_loc]
    concluding_remarks_loc = text.find \
        ('These remarks represent my own views, which do not necessarily represent those of the Federal Reserve Board or the Federal Open Market Committee.')
    if concluding_remarks_loc != -1:
        text = text[:concluding_remarks_loc]
    return text

def clean_speech_text(df):
    """
    Cleans speech text, removing
    urls, links, numbers, references,
    and special characters.

    Parameters:
    df (DataFrame): FRB speech df
        with "full_text" column to
        be cleaned

    Returns:
    DataFrame: pandas DataFrame with
        "full_text" column cleaned
    """
    df_new = df.copy()
    full_text_col = df_new['contents'].apply(lambda x: remove_references(x))
    full_text_col = full_text_col.str.replace('\n', ' ')
    full_text_col = full_text_col.apply \
        (lambda x: re.sub(r'(http)\S+(htm)(l)?', '', x))
    full_text_col = full_text_col.apply(lambda x: re.sub(r'(www.)\S+', '', x))
    full_text_col = full_text_col.apply(lambda x: re.sub(r'[\d]', '', x))
    full_text_col = full_text_col.str.replace('—', ' ')
    full_text_col = full_text_col.str.replace('-', ' ')
    full_text_col = full_text_col.apply(lambda x: re.sub(r'[^\w\s]', '', x))
    full_text_col = full_text_col.apply \
        (lambda x: re.sub(r'([Rr]eturn to text)', '', x))
    full_text_col = full_text_col.apply \
        (lambda x: re.sub(r'([Pp]lay [vV]ideo)', '', x))
    full_text_col = full_text_col.apply \
        (lambda x: x.replace('\n\n[SECTION]\n\n', '').replace('\n', ' ').replace('\r', ' ').strip())
    df_new.drop(labels='contents', axis="columns", inplace=True)
    df_new['contents'] = full_text_col
    return df_new


def get_wordnet_pos(word):
    """
    Maps POS tag to word token
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_speech_text(text):
    """
    Lemmatizes text based on Part
    of Speech (POS) by tokenizing words,
    finding the POS, and passing the
    POS and token into nltk's lemmatizer.

    Parameters:
    text (string): speech text

    Returns:
    list: lemmatized tokens
    """

    lemmatizer = WordNetLemmatizer()
    tokens_lower = [w.lower() for w in nltk.word_tokenize(text)]
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens_lower]

def remove_stop_words(tokens_list):
    """
    Removes English stop words
    from list of tokens

    Parameters:
    tokens_list (list): list of words

    Returns:
    list: list of words sans stop words
    """
    stopwords_without_punct = []
    for word in stopwords.words('english'):
        word = word.replace("'", "")
        stopwords_without_punct.append(word)
    stopped_tokens = [w for w in tokens_list if w not in stopwords_without_punct]
    return [w for w in stopped_tokens if len(w) > 2]


def count_unique_words(text):
    """
    Counts number of unqiue
    words in a piece of text

    Parameters:
    text (string): speech text

    Returns:
    int: number of unique words
    """
    return len(set(text))

# Old function - replaced by lemmatize_speech_text() and remove_stop_words()
# def tokenize_and_remove_stopwords(text):
#     tokens = word_tokenize(text)
#     stopped_tokens = [w for w in tokens if w not in stopwords_without_punct]
#     return stopped_tokens

def get_most_common_words(tokens, num=20):
    """
    Returns list of a number of
    most common tokens (words) in
    a speech

    Parameters:
    tokens (list): list of tokenized
        words from a speech

    num (int): number of top words
        to return

    Returns:
    list of tuples: Words and count
        of the number of times each
        word appears
    """
    fdist = FreqDist(tokens)
    return fdist.most_common(num)

def convert_to_datetime(df):
    """
    Creates 3 new columns in
    FRB speech df, including speech
    date, year, and month.

    Parameters:
    df (DataFrame): FRB speech df
        with "speech_date" column to
        be parsed

    Returns:
    DataFrame: pandas DataFrame with
        3 new date columns
    """
    df_new = df.copy(deep=True)
    df_new['speech_datetime'] = df_new['speech_date'].apply \
        (lambda x: pd.to_datetime(x))
    df_new['speech_year'] = df_new['speech_datetime'].apply(lambda x: x.year)
    df_new['speech_month'] = df_new['speech_datetime'].apply(lambda x: x.month)
    return df_new

def plot_most_common_words(df, article_num=9):
    """
    Plots the 20 most common words in
    a speech, before and after removing
    stop words

    Parameters:
    df (DataFrame): FRB speech
        df with 'common_20_stopped_lemm_words'
        column
    article_num (int): index number
        of the speech for which to
        generate the barplot

    Returns:
    Displays 2 sns barplots of top 20
    words

    """
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle \
        (f"Most common words in Speech: {df.iloc[article_num]['title']}")
    left = fig.add_subplot(121)
    right = fig.add_subplot(122)

    # left subplot without stop words
    sns.barplot \
        (x=[x[0] for x in df.iloc[article_num]['common_20_stopped_lemm_words']],
                y=[x[1] for x in df.iloc[article_num]['common_20_stopped_lemm_words']], ax=left, color='#ffd966'  )  # palette = mycmap)
    left.set_xticklabels(left.get_xticklabels(), rotation=45, horizontalalignment="right")
    left.set_title('Lemmatized Tokens with Stop Words Removed')

    # right subplot with all tokens
    sns.barplot(x=[x[0] for x in df.iloc[article_num]['common_20_lemm_words']],
                y=[x[1] for x in df.iloc[article_num]['common_20_lemm_words']], ax=right, color='gray'  )  # palette = mycmap)
    right.set_xticklabels(right.get_xticklabels(), rotation=45, horizontalalignment="right")
    right.set_title('Lemmatized Tokens')

    plt.show()


def create_dictionary(df, col_name = 'stopped_lemm_words', no_below=10, no_above=0.66, keep_n=10000):
    """
    Creates a dictionary for our corpus

    Parameters:
    df (DataFrame): df containing the
        lemmatized and tokenized corpus

     col_name (str): name of column in
         the df containing the lemmatized
         and tokenized corpus

     no_below (int): Minimum number of documents
         the word mnust appear in to be included
         in the corpus

     no_above (int): Max percentage of documents
         in the corpus the word can appear in.
         Otherwise, word is removed from the corpus

     keep_n (int): Maximum number of
         words to keep in the dictionary

    Returns:
    dictionary: list of tokens in the
        dictionary

    """
    dictionary = corpora.Dictionary(df[col_name])
    print \
        (f"Number of words in dictionary prior to filtering: {len(dictionary)}")
    dictionary.filter_extremes(no_below=10, no_above=0.66, keep_n=10000)
    print(f"Number of words in dictionary after filtering: {len(dictionary)}")
    return dictionary

def create_bow(df, dictionary, col_name = 'stopped_lemm_words'):
    """
    Creates a dictionary for our corpus

    Parameters:
    df (DataFrame): df containing the
        lemmatized and tokenized corpus

     col_name (str): name of column in
         the df containing the lemmatized
         and tokenized corpus

     no_below (int): Minimum number of documents
         the word mnust appear in to be included
         in the corpus

     no_above (int): Max percentage of documents
         in the corpus the word can appear in.
         Otherwise, word is removed from the corpus

     keep_n (int): Maximum number of
         words to keep in the dictionary

    """
    bow_corpus = [dictionary.doc2bow(speech) for speech in df[col_name]]
    return bow_corpus


def get_scores(corpus,
               dictionary,
               df,
               col_name,
               min_num_topics = 2,
               max_num_topics = 15,
               passes=10,
               random_state=100):
    """

    """
    num_topics = list(range(min_num_topics, max_num_topics + 1))
    coherence_scores = []
    perplexity_scores = []

    for num in range(min_num_topics, max_num_topics +1):
        lda_model = gensim.models.LdaMulticore(corpus, num_topics=num, id2word=dictionary,
                                               random_state = random_state, passes = passes)
        perplexity_scores.append(lda_model.log_perplexity(corpus))

        coherence_model_lda = CoherenceModel(model=lda_model, texts=df[col_name], dictionary=dictionary,
                                             coherence='c_v')
        coherence_scores.append(coherence_model_lda.get_coherence())

    data = {'num_topics': num_topics, 'coherence': coherence_scores, 'perplexity': perplexity_scores}
    return pd.DataFrame(data)


def run_and_save_final_lda_model(corpus,
                                 dictionary,
                                 df,
                                 col_name,
                                 num_topics = 11,
                                 passes = 10,
                                 random_state = 100, bow_corpus=None):
    # fit the lda model
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary,
                                           random_state = random_state, passes = passes)

    # pickle the lda model
    pickle.dump(lda_model, open('lda_model' + str(num_topics) + '.sava', 'wb'))

    # create the visualization
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    # save the visualization in html format
    pyLDAvis.save_html(vis, 'lda_' + str(num_topics) + '_topics.html')

    # get the dominant topic information
    df_dominant = get_dominant_topic(lda_model, bow_corpus)
    # pickle the dominant topics
    df_dominant.to_pickle('df_dominant_' + str(num_topics) + '_topics.pkl')

    return lda_model, df_dominant


def get_dominant_topic(lda_model, corpus):
    topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(lda_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                topics_df = topics_df.append(pd.Series
                    ([int(topic_num), round(prop_topic ,4), topic_keywords]), ignore_index=True)
            else:
                break
    topics_df.reset_index(inplace=True)
    topics_df.columns = ['Document_No', 'Dominant_Topic', 'Top_Topic_Perc_Contrib', 'Keywords']

    return topics_df


# EDA

def plot_speeches_per_year(df, figsize = (8, 6), color='#ffd966'):
    fig = plt.figure(figsize = figsize)
    count_by_year = df.groupby('speech_year').count()['index_no'].reset_index()
    sns.barplot(data = count_by_year, x = 'speech_year', y = 'index_no', color = color)
    plt.xticks(rotation=90)
    plt.xlabel('Speech Year', fontsize=14)
    plt.ylabel('Number of Speeches', fontsize=14)
    plt.title('Number of Speeches per Year', fontsize=18)

    plt.show()

def plot_polarity_dist_per_year(df, figsize = (8, 6), color='#ffd966'):
    fig = plt.figure(figsize = figsize)
    sns.boxplot(data=df, x = 'speech_year', y = 'polarity', color = color)
    plt.xticks(rotation=90)
    plt.xlabel('Speech Year', fontsize=14)
    plt.ylabel('Polarity', fontsize=14)
    plt.title('Fed Speech Sentiment (Positive/Negative)', fontsize=18)

    plt.show()

def plot_subjectivity_dist_per_year(df, figsize = (8, 6), color='#ffd966'):
    fig = plt.figure(figsize = figsize)
    sns.boxplot(data=df, x = 'speech_year', y = 'subjectivity', color = color)
    plt.xticks(rotation=90)
    plt.xlabel('Speech Year', fontsize=14)
    plt.ylabel('Subjectivity', fontsize=14)
    plt.title('Fed Speech Subjectivity (Positive/Negative)', fontsize=18)

    plt.show()