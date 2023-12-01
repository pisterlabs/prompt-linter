# library imports
    # webscraping
from selenium import webdriver
import re
import time
    # data analysis
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm_notebook as tqdm
    # natural language processing - NLTK
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet, stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer 
    # natural language processing - Gensim and LDA
import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel
import pyLDAvis.gensim
    # natural language processing - TextBlob (Sentiment)
from textblob import TextBlob
    # data visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def navigate_frb_speeches():
    """
    Navigates the Fed Speeches website 
    and calls get_frb_article_links helper
    function to scrape the urls to all Fed 
    speeches from the Fed website (non-archived
    speeches up until 2006). 
    
    Returns:
    list: Speech urls for all non-archived
    speeches on the Feb website.
    """
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    browser.get("https://www.federalreserve.gov/newsevents/speeches.htm")
    article_urls = []
    new_urls = get_frb_article_links(browser)
    while not article_urls or article_urls[-1] != new_urls[-1]:
        article_urls += get_frb_article_links(browser)
        next_button = browser.find_element_by_css_selector("a[ng-click='selectPage(page + 1, $event)']")
        next_button.click()
        new_urls = get_frb_article_links(browser)
        time.sleep(np.random.randint(5,10))
    browser.close()
    return article_urls
    
def get_frb_article_links(browser):
    """
    Helper function for navigagte_frb_speeches.
    (only works for non-archived speeches)
    
    Parameters:
    browser: Selenium browser instance
    
    Returns:
    list: Speech urls for the current
    page of speeches.
    """
    new_urls = []
    articles = browser.find_elements_by_class_name('itemTitle')
    for article in articles:
        url = article.find_element_by_tag_name('a').get_attribute('href')
        new_urls.append(url)
    return new_urls

def get_frb_speech_text(url_lst):
    """
    Accesses and scrapes all the speech text from a
    list of urls provided. Only works for non-archived
    speeches on the Fed website.
    
    Parameters: 
    url_lst (list): list of speech urls to scrape
    
    Returns:
    list: A list of lists that contains
        the speech url, date, title, speaker, location, 
        and complete text for all speeches in the 
        url_lst.
    """
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    frb_articles = []
    for url in url_lst:
        article_details = []
        article_details.append(url)
        browser.get(url)
        article_times = browser.find_elements_by_class_name('article__time')
        article_details.append(article_times[0].text)
        article_titles = browser.find_elements_by_class_name('title')
        article_details.append(article_titles[0].text)
        article_speakers = browser.find_elements_by_class_name('speaker')
        article_details.append(article_speakers[0].text)
        article_locations = browser.find_elements_by_class_name('location')
        article_details.append(article_locations[0].text)
        article_texts = browser.find_elements_by_xpath('//*[@id="article"]/div[3]')
        article_details.append(article_texts[0].text)
        frb_articles.append(article_details)
        time.sleep(np.random.randint(5,10))
    browser.close()
    return frb_articles

def get_frb_article_links_archived(browser):
    """
    Helper function for navigagte_frb_archived speeches.
    (only works for archived speeches)
    
    Parameters:
    browser: Selenium browser instance
    
    Returns:
    list: Speech urls, titles, speakers
    locations, and dates for the current
    page of speeches.
    """
    new_urls = []
    new_titles = []
    new_speakers = []
    new_locations = []
    new_dates = []
    speeches = browser.find_element_by_id('speechIndex')
    speech_urls = speeches.find_elements_by_tag_name('a')
    for speech in speech_urls:
        url = speech.get_attribute('href')
        new_urls.append(url)
        title = speech.text
        new_titles.append(title)
    speech_dates = speeches.find_elements_by_tag_name('li')
    for speech in speech_dates:
        date_ = re.findall(r'(?<=)(\S+ \d+, \d{4})', speech.text)[0]
        new_dates.append(date_)
    speech_speakers = speeches.find_elements_by_class_name('speaker')
    for speaker in speech_speakers:
        new_speakers.append(speaker.text)
    speech_locations = speeches.find_elements_by_class_name('location')
    for location in speech_locations:
        new_locations.append(location.text)
    return new_urls, new_titles, new_speakers, new_locations, new_dates

def navigate_frb_archived_speeches():
    """
    Navigates the archived Fed Speeches website 
    and calls get_frb_article_links_archiged helper
    function to scrape the urls to all Fed 
    speeches from the Fed website (archived
    speeches up until 1996). 
    
    Returns:
    list: Speech urls for all non-archived
    speeches on the Feb website.
    """
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    browser.get("https://www.federalreserve.gov/newsevents/speech/speeches-archive.htm")
    speech_urls = []
    speakers = []
    locations = []
    dates_ = []
    titles = []
    year_links = []

    list_of_years = browser.find_element_by_xpath('//*[@id="article"]/div/div/div/ul')
    all_year_links = list_of_years.find_elements_by_tag_name("a")
    for year_link in all_year_links:
        url = year_link.get_attribute('href')
        year_links.append(url)
    for url in year_links:
        browser.get(url)
        new_urls, new_titles, new_speakers, new_locations, new_dates = get_frb_article_links_archived(browser)
        speech_urls = speech_urls + new_urls
        titles = titles + new_titles
        speakers = speakers + new_speakers
        locations = locations + new_locations
        dates_ = dates_ + new_dates
        time.sleep(np.random.randint(5,10))
    browser.close()
    # removing extra url accidentally picked up
    del titles[-118]
    del speech_urls[-118]
    return speech_urls, speakers, locations, dates_, titles

def get_frb_speech_text_archived(url_lst):
    """
    Accesses and scrapes all the speech text from a
    list of urls provided. Only works for archived
    speeches on the Fed website.
    
    Parameters: 
    url_lst (list): list of speech urls to scrape
    
    Returns:
    list: speech text
    """
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    speech_text = []
    for url in url_lst:
        browser.get(url)
        paragraphs = browser.find_elements_by_tag_name('p')
        complete_text = ""
        for paragraph in paragraphs:
            complete_text += ' ' + paragraph.text
        speech_text.append(complete_text)
        time.sleep(np.random.randint(5,10))
    browser.close()
    return speech_text


# webscraping functions for FOMC speeches
# project expansion
# not used in current project

def navigate_fomc_speeches():
    fomc_urls = []
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    browser.get("https://www.federalreserve.gov/newsevents/pressreleases.htm")
    new_urls = get_fomc_article_links(browser)
    while not fomc_urls or (not new_urls or fomc_urls[-1] != new_urls[-1]):
        fomc_urls += get_fomc_article_links(browser)
        time.sleep(np.random.randint(5,10))
        next_button = browser.find_element_by_css_selector("a[ng-click='selectPage(page + 1, $event)']")
        next_button.click()
        new_urls = get_fomc_article_links(browser)
    browser.close()
    return fomc_urls

def get_fomc_article_links(browser):
    new_urls = []
    speeches = browser.find_elements_by_class_name('itemTitle')
    for speech in speeches:
        if re.findall(r'FOMC statement', speech.text):
            new_urls.append(speech.find_element_by_tag_name('a').get_attribute('href'))
    return new_urls

def get_fomc_speech_text(url_lst):
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    fomc_speeches = []
    for url in url_lst:
        article_details = []
        article_details.append(url)
        browser.get(url)
        article_times = browser.find_elements_by_class_name('article__time')
        article_details.append(article_times[0].text)
        article_titles = browser.find_elements_by_class_name('title')
        article_details.append(article_titles[0].text)
        article_texts = browser.find_elements_by_xpath('//*[@id="article"]/div[3]')
        article_details.append(article_texts[0].text)
        fomc_speeches.append(article_details)
        time.sleep(np.random.randint(5,10))
    browser.close()
    return fomc_speeches


def navigate_fomc_archived_speeches():
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    browser.get("https://www.federalreserve.gov/newsevents/pressreleases/press-release-archive.htm")
    fomc_urls = []
    titles = []
    year_links = []

    list_of_years = browser.find_element_by_xpath('//*[@id="article"]/div/div/div/ul')
    all_year_links = list_of_years.find_elements_by_tag_name("a")
    for year_link in all_year_links:
        url = year_link.get_attribute('href')
        year_links.append(url)
    for url in year_links:
        browser.get(url)
        new_urls, new_titles = get_fomc_links_archived(browser)
        fomc_urls = fomc_urls + new_urls
        titles = titles + new_titles
        time.sleep(np.random.randint(5,10))
    browser.close()
    return fomc_urls, titles

def get_fomc_links_archived(browser):
    new_urls = []
    new_titles = []
    releases = browser.find_element_by_id('releaseIndex')
    release_urls = releases.find_elements_by_tag_name('a')
    for release in release_urls:
        if re.findall(r'FOMC [Ss]tatement', release.text):
            url = release.get_attribute('href')
            new_urls.append(url)
            title = release.text
            new_titles.append(title)
    return new_urls, new_titles

def get_fomc_text_archived(url_lst):
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    speech_text = []
    fomc_dates = []
    for url in url_lst:
        browser.get(url)
        paragraphs = browser.find_elements_by_tag_name('p')
        complete_text = ""
        for paragraph in paragraphs:
            complete_text += ' ' + paragraph.text
        speech_text.append(complete_text)
        date_ = browser.find_elements_by_tag_name('i')[0]
        date_ = re.findall(r'(?<=[rR]elease [dD]ate: )(\w* \d*,? \d*)', date_.text)[0]
        fomc_dates.append(date_)
        time.sleep(np.random.randint(5,10))
    browser.close()
    return speech_text, fomc_dates

def get_fed_funds_rates(archived=False):
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    if not archived:
        browser.get('https://www.federalreserve.gov/monetarypolicy/openmarket.htm')
    else:
        browser.get('https://www.federalreserve.gov/monetarypolicy/openmarket_archive.htm')
    
    years_txt = []
    years = browser.find_elements_by_tag_name('h4')
    if not archived:
        years = years[1:]
    for year in years:
        years_txt.append(year.text)
    
    dates_ = []
    inc = []
    dec = []
    target = []
    
    rate_tables = browser.find_elements_by_class_name('data-table')
    for i, table in enumerate(rate_tables):
        for j, td in enumerate(table.find_elements_by_tag_name('td')):
            if (j+1) % 4 == 1:
                dates_.append(td.text + ", " + years_txt[i])
            elif (j+1) % 4 == 2:
                inc.append(td.text)
            elif (j+1) % 4 == 3:
                dec.append(td.text)
            elif (j+1) % 4 == 0:
                target.append(td.text)
    browser.close()
    return dates_, inc, dec, target


# Data Cleaning
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
    concluding_remarks_loc = text.find('These remarks represent my own views, which do not necessarily represent those of the Federal Reserve Board or the Federal Open Market Committee.')
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
    full_text_col = df_new['full_text'].apply(lambda x: remove_references(x))
    full_text_col = full_text_col.str.replace('\n', ' ')
    full_text_col = full_text_col.apply(lambda x: re.sub(r'(http)\S+(htm)(l)?', '', x))
    full_text_col = full_text_col.apply(lambda x: re.sub(r'(www.)\S+', '', x))
    full_text_col = full_text_col.apply(lambda x: re.sub(r'[\d]', '', x))
    full_text_col = full_text_col.str.replace('—', ' ')
    full_text_col = full_text_col.str.replace('-', ' ')
    full_text_col = full_text_col.apply(lambda x: re.sub(r'[^\w\s]', '', x))
    full_text_col = full_text_col.apply(lambda x: re.sub(r'([Rr]eturn to text)', '', x))
    full_text_col = full_text_col.apply(lambda x: re.sub(r'([Pp]lay [vV]ideo)', '', x))
    df_new.drop(labels='full_text', axis="columns", inplace=True)
    df_new['full_text'] = full_text_col
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
    df_new['speech_datetime'] = df_new['speech_date'].apply(lambda x: pd.to_datetime(x))
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
    fig.suptitle(f"Most common words in Speech: {df.iloc[article_num]['title']}")
    left = fig.add_subplot(121)
    right = fig.add_subplot(122)
    
    # left subplot without stop words
    sns.barplot(x=[x[0] for x in df.iloc[article_num]['common_20_stopped_lemm_words']],
            y=[x[1] for x in df.iloc[article_num]['common_20_stopped_lemm_words']], ax=left, color='#ffd966')#palette = mycmap)
    left.set_xticklabels(left.get_xticklabels(), rotation=45, horizontalalignment="right")
    left.set_title('Lemmatized Tokens with Stop Words Removed')
    
    # right subplot with all tokens
    sns.barplot(x=[x[0] for x in df.iloc[article_num]['common_20_lemm_words']],
            y=[x[1] for x in df.iloc[article_num]['common_20_lemm_words']], ax=right, color='gray')#palette = mycmap)
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
    print(f"Number of words in dictionary prior to filtering: {len(dictionary)}")
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
    
    for num in range(min_num_topics, max_num_topics+1):
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
                                 random_state = 100):
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
                topics_df = topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
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

