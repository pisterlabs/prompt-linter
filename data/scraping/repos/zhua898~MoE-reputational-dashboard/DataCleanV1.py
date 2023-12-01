import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
import re
from nltk.stem import PorterStemmer
from vader_sentiment.vader_sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from collections import Counter
import openpyxl
import nltk
from gensim import corpora
from gensim.models import LdaModel, Phrases
from nltk.stem import WordNetLemmatizer
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import requests
from bs4 import BeautifulSoup
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect, lang_detect_exception
from langdetect.lang_detect_exception import LangDetectException
from gensim.models import CoherenceModel
from concurrent.futures import ThreadPoolExecutor
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from collections import defaultdict
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix
import torch.nn.functional as torchF
from nltk import ngrams

#ctrl + / = comment
#pandas default UTF-8 and comma as separator
df = pd.read_csv('20230724-Meltwater export.csv', encoding='UTF-16', sep='\t')


#lowercase all content in the report
for col in df.columns:
    if df[col].dtype == 'object':
        #only lowercase the rows where the content is a string
        mask = df[col].apply(type) == str
        df.loc[mask, col] = df.loc[mask, col].str.lower()
print(df.head(10))


#Column: URL
#check if URL is twitter/ non-twitter link
df['is_twitter'] = df['URL'].str.contains('twitter.com')
twitter_count = df['is_twitter'].sum()
non_twitter_count = len(df) - twitter_count
print(twitter_count)
print(non_twitter_count)


#Column: Influencer
#remove @ and replace NaN values with 'NULL'
df['Influencer'] = df['Influencer'].str.replace('@','')
df['Influencer'] = df['Influencer'].fillna('null')
print(df['Influencer'].head(10))


#Column: key phrases
#lowercase all key phrases and replace NaN values with 'NULL'
df['Key Phrases'] = df['Key Phrases'].str.lower()
df['Key Phrases'] = df['Key Phrases'].fillna('NULL')
#print some sample data to check if its replaced with 'NULL'
print(df['Key Phrases'].head(20))


#Column: Tweet Id & Twitter Id
#remove "" and keep only number; replace NaN values with 'NULL'
df['Tweet Id'] = df['Tweet Id'].str.replace('"', '')
df['Twitter Id'] = df['Twitter Id'].str.replace('"', '')
df['Tweet Id'] = df['Tweet Id'].fillna('NULL')
df['Twitter Id'] = df['Twitter Id'].fillna('NULL')
print(df['Tweet Id'].head(20))
print(df['Twitter Id'].head(20))



#Column: URL & User Profile Url
df['URL'] = df['URL'].fillna('NULL')
df['User Profile Url'] = df['User Profile Url'].fillna('NULL')
print(df['User Profile Url'].head(10))

#use regex tp replace youtube links in the hit sentence column with NULL
pattern = r'https?://(www\.)?youtube(\.com|\.be)/'
df.loc[df['URL'].str.contains(pattern, na=False, regex=True), 'Hit Sentence'] = "NULL"


#Sheffin
#column: Hit Sentence
#firstly replace NaN values with 'null'
df['Hit Sentence'] = df['Hit Sentence'].fillna('NULL')

session = requests.Session()
retry = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

def fetch_content(url):
    print(f"Fetching {url}...")
    try:
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return url, ' '.join(paragraphs)
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return url, ""

df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%Y %I:%M%p")
grouped = df.groupby([df['Date'].dt.year, df['Date'].dt.month])

for (year, month), group in grouped:
    try:
        print(f"Processing articles from {month}-{year}...")
        urls = group['URL'].tolist()
        non_twitter_urls = [url for url in urls if "twitter.com" not in url]

        with ThreadPoolExecutor(max_workers=1000) as executor:
            news_sentences = list(executor.map(fetch_content, non_twitter_urls))

        url_content_dict = {url: content for url, content in news_sentences}
        group['web_content'] = group['URL'].map(url_content_dict)
        df.loc[group.index, 'web_content'] = group['web_content']

    except Exception as e:
        print(f"Error processing data for {month}-{year}: {e}")

df['raw_combined'] = df.apply(lambda row: row['web_content'] if row['Hit Sentence'] == 'NULL' else row['Hit Sentence'], axis=1)
df['raw_combined'] = df['raw_combined'].replace('', 'NULL')
df['raw_combined'] = df['raw_combined'].str.lower()
df['raw_combined'] = df['raw_combined'].fillna('NULL')


def ml_preprocessing(text):

    if not isinstance(text, str):
        return ''

    #remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    #remove digits
    text = re.sub(r'\d+', '', text)

    #remove URLs
    text = re.sub(r'http\S+', '', text)

    #Remove Twitter mentions
    text = re.sub(r'@\w+', '', text)

    substrings_to_remove = ['rt', 'amp', 're', 'qt']

    # Iterate through substrings and remove them if they appear at the start of the text
    for substring in substrings_to_remove:
        if text.startswith(substring):
            text = text[len(substring):]

    #remove additional special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    #remove unwanted spaces
    text = re.sub(r'\s+', ' ', str(text)).strip()

    #remove non-ASCII characters
    text = ''.join(character for character in text if ord(character) < 128)

    return text.strip()



df['processed_combined'] = df['raw_combined'].apply(ml_preprocessing)

df.to_excel('raw_content.xlsx',index=False)



#Sheffin: machine learning
df = pd.read_excel('raw_content.xlsx')

df.dropna(subset=['processed_combined'], inplace=True)
text = df['processed_combined']

def remove_unwanted_spaces(text):
    cleaned_text = re.sub(r'\s+', ' ', str(text)).strip()
    return cleaned_text

df["processed_combined"] = df["processed_combined"].apply(remove_unwanted_spaces)

# Define the date range
# start_date = pd.to_datetime('2022-01-01 00:00:00')
# end_date = pd.to_datetime('2022-09-19 23:59:59')
#
# # Filter the DataFrame for dates within the specified range
# df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
#
# df['Sentiment'].value_counts().plot(kind='bar')
# plt.show()
def filter_dataframe_by_date(df, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

#df_filtered = filter_dataframe_by_date(df, '2022-01-01 00:00:00', '2022-09-19 23:59:59')


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def analyze_reviews(text):
    review_text = str(text)[:512]
    tokens = tokenizer.encode(review_text, return_tensors='pt')
    result = model(tokens)
    probabilities = torchF.softmax(result.logits, dim=-1)
    score = int(torch.argmax(probabilities)) + 1
    if score > 3:
        sentiment = "positive"
    elif score < 3:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return sentiment

df["sentiment_result"] = df["processed_combined"].apply(analyze_reviews)

class_labels = ["positive", "neutral", "negative", "not rated"]
actual = df['Sentiment']
predicted = df['sentiment_result']

def plot(actual, predicted, class_labels):
    conf_matrix = confusion_matrix(actual, predicted, labels=class_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks(np.arange(len(class_labels)) + 0.5, class_labels)
    plt.yticks(np.arange(len(class_labels)) + 0.5, class_labels)
    plt.show()

#plot(actual, predicted, class_labels)


#Topic filtering
df['is_twitter'] = df['URL'].str.contains('twitter.com')
def create_review_column(df):
    df['review'] = df.apply(lambda row: row['processed_combined'] if row['is_twitter'] else row['Headline'], axis=1)
    return df

df = create_review_column(df)

# Define the keywords for each class
classes = {
    'Equity': ['racism', 'Pacific education', 'Māori education', 'Māori medium', 'fair', 'kaupapa Māori','Kōhanga','kura',
               'wānanga','immersion','learning support','migrant','culturally and linguistically diverse',
              'CALD','te reo','equity','fair','inequity','digital divide','disadvantaged','barriers to education'],
    'Achievement': ['academic performance', 'NCEA', 'certificate','scholarship','qualification', 'tournament','competition',
                   'achievement','OECD'],
    'Attendance': ['attendance', 'truancy', 'unjustified absence','All in for Education','skipping school','truant',
                   'wagging','engagement'],
    'Workforce': ['workforce', 'teacher supply', 'teacher pay', 'pay equity', 'education wellbeing','negotiation',
                 'strike','teacher training','PPTA',' pay parity',' teacher shortage','educator shortage','educator supply',
                 'educator pay','certified','collective agreement',' industrial action'],
    'Wellbeing': ['wellbeing', 'mental health', 'bullying', 'pastoral care','safety','school lunches','"Ka Ora, Ka Ako"',
                 'covid','pandemic','sick','health'],
    'Curriculum': ['curriculum', 'Te Marautanga', 'sex education', 'science education', 'literacy', 'numeracy'],
    'Te Mahau': ['Tomorrow’s', 'Te Mahau', 'Redesigned Ministry', 'Te Poutāhū', 'curriculum centre', 'regional office',
                'local office']
}

reviews = df['processed_combined'].astype(str).tolist()
reviews = df['review'].astype(str).tolist()

def compile_keywords(classes):
    compiled_classes = {}
    for class_name, keywords in classes.items():
        compiled_classes[class_name] = set(keywords)
    return compiled_classes

def get_ngrams(text, n):
    tokens = text.lower().split()
    ngram_list = [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return ngram_list

def classify_review(review, classes):

    tokens = review.lower().split()

    scores = {class_name: 0 for class_name in classes}

    for n in range(1, 4):
        ngrams_list = get_ngrams(review, n)
        for token in ngrams_list:
            for class_name, keywords in classes.items():
                scores[class_name] += sum(1 for keyword in keywords if keyword in token)

    max_score = max(scores.values())
    if max_score == 0:
        return 'Undefined'

    best_class = max(scores, key=scores.get)
    return best_class


# Precompile the keyword sets
compiled_classes = compile_keywords(classes)


df['Topic'] = df['processed_combined'].apply(lambda x: classify_review(x, compiled_classes))

cluster_counts = df['Topic'].value_counts()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')


for i, count in enumerate(cluster_counts.values):
    ax.text(i, count + 0.1, str(count), ha='center', va='bottom')

plt.title('Number of Reviews in Each Cluster')
plt.xlabel('Cluster Class')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45, ha='right')
#plt.show()

def group_sentiments_by_cluster(df):
    grouped_df = df.groupby(['Topic', 'sentiment_result']).size().reset_index(name='count')
    pivoted_df = grouped_df.pivot(index='Topic', columns='sentiment_result', values='count').reset_index()
    pivoted_df = pivoted_df.fillna(0)
    pivoted_df.columns = [f'Topic_{col}' if col != 'Topic' else col for col in pivoted_df.columns]
    merged_df = pd.merge(df, pivoted_df, on='Topic', how='left')
    return merged_df

result_df = group_sentiments_by_cluster(df)

result_df.to_excel('demo_1104.xlsx', index=False)











#Unsupervised learning:
#phrasal verb
ps = PorterStemmer()

# remove stop words, punctuation, and numbers or digits from the Hit sentence column
def lda_process_text(text):
    #remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    #remove digits
    text = re.sub(r'\d+', '', text)

    #remove URLs
    text = re.sub(r'http\S+', '', text)

    #Remove Twitter mentions
    text = re.sub(r'@\w+', '', text)

    substrings_to_remove = ['rt', 'amp', 're', 'qt']

    # Iterate through substrings and remove them if they appear at the start of the text
    for substring in substrings_to_remove:
        if text.startswith(substring):
            text = text[len(substring):]

    #remove additional special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # remove unwanted spaces
    text = re.sub(r'\s+', ' ', str(text)).strip()

    #remove non-ASCII characters
    text = ''.join(character for character in text if ord(character) < 128)

    return text.strip()

df['Hit Sentence'] = df['Hit Sentence'].apply(lda_process_text)



#8/18
#LDA
nltk.download('stopwords')
nltk.download('wordnet')

#9/12 add lda top keywords to columns
tf_dict = {}
keywords_dict = {}
frequency_dict = {}
rows = []

df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y %I:%M%p')
df['Month-Year'] = df['Date'].dt.to_period('M')

lemmatizer = WordNetLemmatizer()

#exclude useless words
excluded_words = {'stated', 'going', 'null', "said", "would", "also", "one", "education", "school", "children",
                  "ministry", "sector", "teacher", "teachers", "government", "schools", "kids", "home", "students",
                  "classes", "parents", "child", "staff", "families", "person", "percent", "work", "rain",
                  "year", "year,", "years.", "since", "last", "group", "whether", "asked", "new", "zealand", "say", "search",
                  "people", "way", "time", "point", "thing", "part", "something", "student", "te", "name", "m", "use",
                  "say", "made", "month", "day", "moe", "years", "years.", "years,", "e", "http",
                  "havent", "like", "need", "every", "know", "wrote", "make", "get", "need", "think", "put",
                  "e", "купить", "don't", "need", "get"
            }

stop_words = set(stopwords.words('english')).union(excluded_words)

for month_year, group in df.groupby('Month-Year'):
    #tokenize, remove stopwords, lemmatize and filter non-alpha tokens
    sentences = [nltk.word_tokenize(sent.lower()) for sent in group['Hit Sentence']]
    cleaned_sentences = [
        [lemmatizer.lemmatize(token) for token in sentence if token not in stop_words and token.isalpha() and len(token) > 2]
        for sentence in sentences
    ]

    #8/25 change
    #list possible combination of 2/3 common words
    bigram_model = Phrases(cleaned_sentences, min_count=5, threshold=100)
    trigram_model = Phrases(bigram_model[cleaned_sentences], threshold=100)
    tokens_with_bigrams = [bigram_model[sent] for sent in cleaned_sentences]
    tokens_with_trigrams = [trigram_model[bigram_model[sent]] for sent in tokens_with_bigrams]

    #flatten list of sentences for LDA
    all_tokens = [token for sentence in tokens_with_trigrams for token in sentence]

    # Calculate term frequency for this month-year and store in the dictionary
    tf_dict[month_year] = Counter(all_tokens)

    # Get top 30 keywords for this month-year
    keywords_dict[month_year] = [keyword for keyword, freq in tf_dict[month_year].most_common(30)]

    # Get frequencies for the top 30 keywords
    top_30_keywords = [keyword for keyword, freq in tf_dict[month_year].most_common(30)]
    top_30_frequencies = [str(tf_dict[month_year][keyword]) for keyword in top_30_keywords]

    for keyword, frequency in zip(top_30_keywords, top_30_frequencies):
        rows.append({'Month-Year': str(month_year), 'Keyword': keyword, 'Frequency': frequency})

    # Create the new dataframe
    keywords_df = pd.DataFrame(rows)
    df['Month-Year'] = df['Month-Year'].astype(str)
    keywords_df['Month-Year'] = keywords_df['Month-Year'].astype(str)


    #corpus for LDA
    dictionary = corpora.Dictionary([all_tokens])
    corpus = [dictionary.doc2bow(text) for text in [all_tokens]]

    #LDA implementation
    num_topics = 3
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    topics = lda.print_topics(num_words=30)
    month_keywords = []
    for topic in topics:
        _, words = topic
        keywords = ' '.join(word.split('*')[1].replace('"', '').strip() for word in words.split('+'))
        month_keywords.append(keywords)  # Accumulate keywords for this topic
        print(f"Month-Year: {month_year}")
        print(topic)


    #display relevant terms
    lda_display = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)
    filename = f'ldaTweet_{month_year}.html'
    pyLDAvis.save_html(lda_display, filename)


    lda_df = pd.DataFrame(rows)
    lda_df.to_excel("Tweet_LDA_output.xlsx", index=False)




#WEB SCRAPING for non-twitter content: news websites and articles
#initialize stemmer
stemmer = PorterStemmer()
session = requests.Session()
retry = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

def fetch_content(url):
    print(f"Fetching {url}...")
    try:
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return ' '.join(paragraphs)
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return ""

# Tokenization
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
expanded_stopwords = set(stopwords.words('english')).union({'stated', 'going', 'null', "said", "would", "also", "one", "education", "school", "children",
                  "ministry", "sector", "teacher", "teachers", "government", "schools", "kids", "home", "students",
                  "classes", "parents", "child", "staff", "families", "person", "percent", "work", "rain",
                  "year", "year,", "years.", "since", "last", "group", "whether", "asked", "new", "zealand", "say", "search",
                  "people", "way", "time", "point", "thing", "part", "something", "student", "te", "name", "m", "use",
                  "say", "made", "month", "day", "moe", "years", "years.", "years,", "e", "http",
                  "havent", "like", "need", "every", "know", "wrote", "make", "get", "need", "think", "put",
                  "e", "купить", "don't", "need", "get"

                                                            })

#convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%Y %I:%M%p")
grouped = df.groupby([df['Date'].dt.year, df['Date'].dt.month])
all_documents = []


all_rows = []
for(year, month), group in grouped:
    try:
        print(f"Processing articles from {month}-{year}...")
        # URLs
        urls = group['URL'].tolist()
        # Filter out Twitter URLs
        non_twitter_urls = [url for url in urls if "twitter.com" not in url]

        #Important: max_workers significantly affect the speed of web scraping
        with ThreadPoolExecutor(max_workers=1000) as executor:
            news_sentences = list(executor.map(fetch_content, non_twitter_urls))

        # Filter out non-English content
        english_news = []
        for news in news_sentences:
            try:
                if detect(news) == 'en':
                    english_news.append(news)
            except LangDetectException:
                pass

        documents = []
        for sentence in english_news:
            #Lemmatization
            tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(sentence.lower()) if token not in expanded_stopwords and token.isalpha()]
            #Stemming
            #tokens = [stemmer.stem(token) for token in word_tokenize(sentence.lower()) if token not in expanded_stopwords and token.isalpha()]
            # Consider keeping only nouns for better topic clarity (requires POS tagging)
            tokens = [token for token, pos in nltk.pos_tag(tokens) if pos.startswith('NN')]
            documents.append(tokens)

        #Combine possible words using bigrams and trigrams
        bigram_model_website = Phrases(documents, min_count=5, threshold=100)
        trigram_model_website = Phrases(bigram_model_website[documents], threshold=100)
        documents_with_bigrams = [bigram_model_website[doc] for doc in documents]
        documents_with_trigrams = [trigram_model_website[bigram_model_website[doc]] for doc in documents_with_bigrams]

        # Create LDA model for this month
        # dictioary to store raw term frequencies across all documents
        term_frequencies = defaultdict(int)
        for document in documents_with_trigrams:
            for term in document:
                term_frequencies[term] += 1
        dictionary = Dictionary(documents_with_trigrams)
        corpus = [dictionary.doc2bow(text) for text in documents_with_trigrams]
        lda = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

        topics = lda.print_topics(num_words=10)
        for topic_num, topic in topics:
            pairs = topic.split('+')
            for pair in pairs:
                weight, word = pair.split('*')
                word = word.replace('"', '').strip()
                all_rows.append({
                    'Month-Year': f"{month}-{year}",
                    'Keyword': word,
                    'Weight': float(weight),
                    'Raw Frequency': term_frequencies[word]
                })
            print(topic)

        # Generate LDA visualization for this month and save to an HTML file
        lda_display = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)
        html_filename = f'ldaWeb_{year}_{month}.html'
        pyLDAvis.save_html(lda_display, html_filename)

    except Exception as e:
        print(f"Error processing data for {month}-{year}: {e}")

#saving web scraping content to excel
keywords_df = pd.DataFrame(all_rows)
keywords_df.to_excel("web_LDA_output.xlsx", index=False)







#because csv would change ID to scientific notation, the format is changed to xlsx for the output
df.to_excel('demo_1104.xlsx',index=False)









# Define the data

df = pd.read_excel('20230910-Public Sector Reputation index.xlsx', header=None)
# Find the row indices where the headings are located
headings = [
    r'Reputation Score \(out of 100\)',
    r'Reputation Score \(out of 100\) by the 4 Pillars',
    r'% agree with the following statement \(rates 5 to 7 out of 7\)'
]

heading_indices = df[df.apply(lambda row: any(re.match(pattern, str(val)) for pattern in headings for val in row), axis=1)].index
# Split the DataFrame into three separate DataFrames based on the headings
dfs = []
for i in range(len(heading_indices) - 1):
    start_index = heading_indices[i]
    end_index = heading_indices[i + 1]
    sub_df = df[start_index:end_index]
    dfs.append(sub_df)

# Add the last section
dfs.append(df[heading_indices[-1]:])

# Write each section to a separate Excel file
for i, sub_df in enumerate(dfs):
    sub_df.to_excel(f'~/Downloads/Kantar_Data_{i}.xlsx', index=False)
























