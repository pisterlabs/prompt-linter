#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')
import os.path
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
nltk.download('wordnet')
import pandas as pd
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import gensim
import string


# In[ ]:


# Link extraction
get_ipython().system('pip install python-edgar')


# In[ ]:


import edgar
edgar.download_index('/Users/maksymlesyk/extract', 2017, "MyEDGARScraper/v1.0", skip_all_present_except_last=False)


# In[ ]:


import os
import pandas as pd

# Folder containing the .tsv files
folder_path = '/Users/maksymlesyk/extract'

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tsv'):
        file_path = os.path.join(folder_path, filename)
        
        # Read the current .tsv file into a DataFrame
        current_df = pd.read_csv(file_path, sep='|', header=None,
                                 names=['cik', 'conm', 'type', 'date', 'txt_file', 'html_file'])
        
        # Append the current DataFrame to the list of DataFrames
        dataframes.append(current_df)

# Concatenate all DataFrames in the list into a single DataFrame
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame to a new .tsv file
merged_file_path = '/Users/maksymlesyk/extract/merged_file.tsv'
merged_df.to_csv(merged_file_path, sep='|', index=False)

print(f'Merged data saved to {merged_file_path}')


# In[ ]:


# Filter
filtered_df = merged_df[merged_df['type'] == '10-K']


# In[ ]:


# Add prefix
filtered_df['Link'] = 'https://www.sec.gov/Archives/'+ filtered_df['html_file']


# In[ ]:


import re
pattern1 = r'-(\d{2})-'


# In[ ]:


filtered_df.loc[:, 'date'] = pd.to_datetime(filtered_df['date'])


# In[ ]:


filtered_df['txt_file'].apply(lambda x: re.findall(pattern1, x)).explode().unique()


# In[ ]:


filtered_df['year_report']='20'+filtered_df['txt_file'].apply(lambda x: re.findall(pattern1, x)).explode()
filtered_df.loc[:, 'year_report'] = filtered_df['year_report'].astype(int)


# In[ ]:


import csv
import random
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.service import Service as ServiceBase

# Specify the path to your ChromeDriver executable
chrome_driver_path = '/usr/local/bin/chromedriver'

# Create an empty column "Link_htm" to store the extracted links
filtered_df['Link_htm'] = None

for index, row in trial_df.iterrows():
    print('Start fetching URL:', row['Link'], '...')

    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    # Create a new Chrome service with the custom executable path
    chrome_service = Service(chrome_driver_path)

    # Create Chrome options with the user-agent
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36")

    # Use the custom Chrome service and options
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

    try:
        print('Before driver.get()')  # Add this line for debugging
        driver.get(row['Link'])
        print('After driver.get()')  # Add this line for debugging
        time.sleep(3 + random.random() * 3)

        # Locate the link using the provided XPath
        link_element = driver.find_element(By.XPATH, '//*[@id="formDiv"]/div/table/tbody/tr[2]/td[3]/a')

        # Get the link's href attribute
        link_text = link_element.get_attribute("href")

        # Store the link in the "Link_htm" column
        filtered_df.at[index, 'Link_htm'] = link_text

        end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('Success!', start_time, ' --> ', end_time, '\n')

    except Exception as e:
        end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('Error!', start_time, ' --> ', end_time, '\n')

    driver.quit()


# In[ ]:


tt_df=filtered_df


# In[1]:


import pandas as pd
# Read the TSV file using pd.read_csv
tt_df = pd.read_csv("/Users/maksymlesyk/new_file2.csv")
# Inspect the contents of the DataFrame
print(tt_df)


# In[2]:


#!/usr/bin/env python
# coding: utf-8

# # 10-K form
# ## Business, Risk, and MD&A
# The function *parse_10k_filing()* parses 10-K forms to extract the following sections: business description, business risk, and management discussioin and analysis. The function takes two arguments, a link and a number indicating the section, and returns a list with the requested sections. Current options are **0(All), 1(Business), 2(Risk), 4(MDA).**
# 
# Caveats:
# The function *parse_10k_filing()* is a parser. You need to feed a SEC text link into it. There are many python and r packages to get a direct link to the fillings.
# 

import re
import unicodedata
from bs4 import BeautifulSoup as bs
import requests
import sys

def parse_10k_filing(link, section):
    
    if section not in [0, 1, 2, 3]:
        print("Not a valid section")
        sys.exit()
    
    def get_text(link):
        page = requests.get(link, headers={'User-Agent': 'Mozilla'})
        html = bs(page.content, "lxml")
        text = html.get_text()
        text = unicodedata.normalize("NFKD", text).encode('ascii', 'ignore').decode('utf8')
        text = text.split("\n")
        text = " ".join(text)
        return(text)
    
    def extract_text(text, item_start, item_end):
        item_start = item_start
        item_end = item_end
        starts = [i.start() for i in item_start.finditer(text)]
        ends = [i.start() for i in item_end.finditer(text)]
        positions = list()
        for s in starts:
            control = 0
            for e in ends:
                if control == 0:
                    if s < e:
                        control = 1
                        positions.append([s,e])
        item_length = 0
        item_position = list()
        for p in positions:
            if (p[1]-p[0]) > item_length:
                item_length = p[1]-p[0]
                item_position = p

        item_text = text[item_position[0]:item_position[1]]

        return(item_text)

    text = get_text(link)
        
    if section == 1 or section == 0:
        try:
            item1_start = re.compile("item\s*[1][\.\;\:\-\_]*\s*\\b", re.IGNORECASE)
            item1_end = re.compile("item\s*1a[\.\;\:\-\_]\s*Risk|item\s*2[\.\,\;\:\-\_]\s*Prop", re.IGNORECASE)
            businessText = extract_text(text, item1_start, item1_end)
        except:
            businessText = "Something went wrong!"
        
    if section == 2 or section == 0:
        try:
            item1a_start = re.compile("(?<!,\s)item\s*1a[\.\;\:\-\_]\s*Risk", re.IGNORECASE)
            item1a_end = re.compile("item\s*2[\.\;\:\-\_]\s*Prop|item\s*[1][\.\;\:\-\_]*\s*\\b", re.IGNORECASE)
            riskText = extract_text(text, item1a_start, item1a_end)
        except:
            riskText = "Something went wrong!"
            
    if section == 3 or section == 0:
        try:
            item7_start = re.compile("item\s*[7][\.\;\:\-\_]*\s*\\bM", re.IGNORECASE)
            item7_end = re.compile("item\s*7a[\.\;\:\-\_]\sQuanti|item\s*8[\.\,\;\:\-\_]\s*", re.IGNORECASE)
            mdaText = extract_text(text, item7_start, item7_end)
        except:
            mdaText = "Something went wrong!"
    
    if section == 0:
        data = [businessText, riskText, mdaText]
    elif section == 1:
        data = [businessText]
    elif section == 2:
        data = [riskText]
    elif section == 3:
        data = [mdaText]
    return data


# In[3]:


tt_df['Link_htm'] = tt_df['Link_htm'].str.replace('/ix?doc=', '')


# In[4]:


tt_df=tt_df[tt_df['Link_htm'].notna()]


# In[5]:


tt_df = tt_df.head(5000)


# In[6]:


import pandas as pd
from tqdm import tqdm

# Assuming you have a DataFrame named trial_df1 with a "Link_htm" column
# Create an empty "Text" column
tt_df['Text'] = ""

# Define the function to apply to each row
def apply_parse_10k_filing(link):
    # Call the parse_10k_filing function with section 2 (Risk) and return the Risk section
    return parse_10k_filing(link, 2)

# Create a progress bar
t = tqdm(total=len(tt_df))

# Apply the function to each row in the "Link_htm" column and store the result in the "Text" column
for index, row in tt_df.iterrows():
    tt_df.at[index, 'Text'] = apply_parse_10k_filing(row['Link_htm'])
    t.update(1)  # Update the progress bar

# Close the progress bar
t.close()

# Display the first few rows of the DataFrame to verify the results
tt_df


# In[2]:


import pandas as pd
# Read the TSV file using pd.read_csv
tt_df = pd.read_csv("/Users/maksymlesyk/5000rows.csv", sep='|')
# Inspect the contents of the DataFrame
print(tt_df)


# In[3]:


tt_df['tag'] = tt_df['Text'].apply(lambda x: 1 if x != ['Something went wrong!'] else 0)


# In[4]:


# Delete rows with tag value equal to 0
tt_df = tt_df[tt_df['tag'] != 0]


# In[5]:


tt_df['Text']=tt_df['Text'].astype(str)


# In[6]:


tt_df['Text'] = tt_df['Text'].str[2:-2]


# In[13]:


tt_df


# In[7]:


get_ipython().system(' pip install textstat')
import pandas as pd
import textstat


# In[8]:


def calculate_readability_scores(text):
    try:
        # You can use various readability formulas provided by textstat.
        # For example, here, we calculate Flesch-Kincaid Grade Level and Automated Readability Index.
        flesch_kincaid = textstat.flesch_kincaid_grade(text)
        ari = textstat.automated_readability_index(text)
        smog = textstat.smog_index(text)
        return flesch_kincaid, ari, smog
    except Exception as e:
        return None, None


# In[9]:


tt_df[['Flesch-Kincaid Grade Level', 'Automated Readability Index', 'SMOG Index']] = tt_df['Text'].apply(calculate_readability_scores).apply(pd.Series)


# In[10]:


tt_df=tt_df[tt_df['SMOG Index']!=0]


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Define the step for the histogram
step = 0.1

# Create subplots with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Define readability measures
readability_measures = ['Flesch-Kincaid Grade Level', 'SMOG Index', 'Automated Readability Index']

for i, measure in enumerate(readability_measures):
    # Calculate the frequencies for the current readability measure
    frequencies = tt_df[measure].value_counts(bins=int((tt_df[measure].max() - tt_df[measure].min()) / step))

    # Sort the frequencies by index (bin)
    frequencies = frequencies.sort_index()

    # Calculate percentiles (e.g., 25th, 50th, and 75th percentiles)
    percentiles = [25, 50, 75]
    percentile_values = np.percentile(tt_df[measure], percentiles)

    # Calculate the maximum frequency for the current subplot
    max_frequency = frequencies.max()

    # Create a histogram plot in the current subplot
    axes[i].bar(frequencies.index.left, frequencies, width=step, align='edge')
    axes[i].set_xlabel(f'{measure} Bins')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Histogram of {measure}')

    # Add vertical lines for percentiles
    for percentile_value in percentile_values:
        # Find the closest index in the frequency index
        idx = (np.abs(frequencies.index.left - percentile_value)).argmin()
        x_position = frequencies.index.left[idx]
        y_position = max_frequency  # Set the y_position to the maximum frequency value
        axes[i].axvline(x=x_position, color='red', linestyle='--')
        axes[i].text(x_position, y_position, f'{percentile_value:.1f}', color='black', ha='center', va='bottom', fontsize=6)

# Adjust subplot spacing
plt.tight_layout()
plt.show()


# In[11]:


# Define a function to count words in a text
def count_words(text):
    words = text.split()
    return len(words)

# Apply the function to the "Text" column and create a new column "WordCount"
tt_df['WordCount'] = tt_df['Text'].apply(count_words)


# In[12]:


len(tt_df[tt_df['WordCount']<1000])


# In[13]:


tt_df=tt_df[tt_df['WordCount']>1000]


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create a density line plot
plt.figure(figsize=(10, 6))  # Set the figure size
sns.kdeplot(tt_df['WordCount'], shade=True, cut=0)  # Create a kernel density estimate plot, cut at 0

# Set labels and title
plt.xlabel('Word Count')
plt.ylabel('Density')
plt.title('Density Line Plot of Word Count')

# Show the plot
plt.show()


# In[23]:


import matplotlib.pyplot as plt

# Create a histogram of Word Count frequencies
plt.figure(figsize=(10, 6))  # Set the figure size
plt.hist(tt_df['WordCount'], bins=range(0, 70001, 1000), edgecolor='k', alpha=0.7)

# Set x-axis and y-axis labels
plt.xlabel('Word Count')
plt.ylabel('Frequency (Number of Documents)')

# Set title
plt.title('Word Count Frequency Histogram')

# Show the plot
plt.show()


# In[14]:


import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

exclude=['million','billion','january','february','march','april','may','june','july','august','september','october','november','december']

with open('/Users/maksymlesyk/Downloads/StopWords_GenericLong.txt', 'r') as file:
    stop_words = set(word.strip() for word in file)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    words = word_tokenize(text)
    
    # Filter out words in the 'exclude' list
    words = [word for word in words if word.lower() not in exclude]

    # Remove stopwords
    words = [word for word in words if word.lower() not in stop_words]

    # Remove punctuation and digits
    #words = [word for word in words if word not in string.punctuation and not word.isdigit()]
    words = [''.join(char for char in word if char not in string.punctuation and not char.isdigit()) for word in words]

    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]

    # Create the tokenized Clean_text
    return words

tt_df['Clean_text'] = tt_df['Text'].apply(preprocess_text)


# In[15]:


with open('/Users/maksymlesyk/Downloads/StopWords_GenericLong.txt', 'r') as file:
    stop_words = set(word.strip() for word in file)

lemmatizer = WordNetLemmatizer()

def preprocess_non_tokenized_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

    # Remove punctuation and digits
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])

    # Lemmatize without tokenization
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    # Create the non-tokenized NT_Clean_text
    return text

# Assuming you have a DataFrame named tt_df

tt_df['NT_Clean_text'] = tt_df['Text'].apply(preprocess_non_tokenized_text)


# In[16]:


from collections import Counter

# Concatenate tokens for each document
document_tokens = tt_df['Clean_text'].apply(lambda tokens: ' '.join(tokens))

# Combine all document tokens into a single string
all_tokens = ' '.join(document_tokens)

# Split the combined string into tokens
all_tokens_list = all_tokens.split()

# Create a bag of words (BoW) using Counter
bow = Counter(all_tokens_list)

# The 'bow' variable now contains the frequency of each token across all documents


# In[27]:


bow_simple_df = pd.DataFrame(list(bow.items()), columns=['Words', 'Count'])

bow_simple_df1=bow_simple_df.sort_values(by='Count',ascending=False).head(25)

plt.figure(figsize=(15, 10))
bow_simple_df1.plot.bar(x='Words', y='Count')
plt.xticks(rotation=50)
plt.show()


# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Convert the Counter to a list of tokens
tokens_list = list(bow.elements())

# Create a TF-IDF vectorizer with bigrams (n-grams of size 2)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))

# Fit and transform the BoW tokens to TF-IDF representation
tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(tokens_list)])

# Now, tfidf_matrix contains the TF-IDF representation of bigrams
# Get the bigrams: bigrams
bigrams = tfidf_vectorizer.get_feature_names_out()


# In[29]:


tfidf_df_gram = pd.DataFrame({'word': bigrams, 'tfidf_score': tfidf_matrix.sum(axis=0).A1})

# Sort the DataFrame by TF-IDF scores in descending order to get the most important words
most_important_bigrams = tfidf_df_gram.sort_values(by='tfidf_score', ascending=False)

# Print the most important words
most_important_bigrams.head(20)


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer with bigrams (n-grams of size 2)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2))

# Fit and transform the concatenated tokens to TF-IDF representation
tfidf_matrix = tfidf_vectorizer.fit_transform(document_tokens)

# Now, tfidf_matrix contains the TF-IDF representation of bigrams


# In[31]:


bigrams = tfidf_vectorizer.get_feature_names_out()


# In[32]:


tfidf_df_gram = pd.DataFrame({'word': bigrams, 'tfidf_score': tfidf_matrix.sum(axis=0).A1})

# Sort the DataFrame by TF-IDF scores in descending order to get the most important words
most_important_bigrams = tfidf_df_gram.sort_values(by='tfidf_score', ascending=False)

# Print the most important words
most_important_bigrams.head(20)


# In[33]:


from nltk import bigrams, FreqDist

# Combine all document tokens into a single list of tokens
all_tokens = [token for tokens in tt_df['Clean_text'] for token in tokens]

# Generate bigrams from the combined list of tokens
all_bigrams = list(bigrams(all_tokens))

# Calculate the frequency of each bigram
bigram_freq = FreqDist(all_bigrams)

# Find bigrams that are most common with the word 'risk'
target_word = 'risk'
common_bigrams = [bigram for bigram, freq in bigram_freq.items() if target_word in bigram and freq > 1]

# Sort common_bigrams by frequency in descending order
common_bigrams.sort(key=lambda bigram: -bigram_freq[bigram])

# Display the top N common bigrams (adjust N as needed)
N = 10
for bigram in common_bigrams[:N]:
    print(bigram)


# In[34]:


from nltk import bigrams, FreqDist, pos_tag
from nltk.corpus import wordnet
import string

# Define a function to get the WordNet POS tag for a given tag
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    else:
        return ''

# Combine all document tokens into a single list of tokens
all_tokens = [token for tokens in tt_df['Clean_text'] for token in tokens]

# Generate bigrams from the combined list of tokens
all_bigrams = list(bigrams(all_tokens))

# Calculate the frequency of each bigram
bigram_freq = FreqDist(all_bigrams)

# Find bigrams that include the word 'risk' and a word tagged as a noun or adjective
target_word = 'risk'
common_bigrams = []

# Perform part-of-speech tagging and filter bigrams
for bigram, freq in bigram_freq.items():
    words = list(bigram)
    tagged_words = pos_tag(words)

    # Check if 'risk' is in the bigram and one of the words is tagged as a noun or adjective
    if target_word in bigram and any(get_wordnet_pos(tag) in [wordnet.ADJ, wordnet.NOUN] for word, tag in tagged_words):
        common_bigrams.append(bigram)

# Sort common_bigrams by frequency in descending order
common_bigrams.sort(key=lambda bigram: -bigram_freq[bigram])

# Display the top N common bigrams (adjust N as needed)
N = 10
for bigram in common_bigrams[:N]:
    print(bigram)


# In[37]:


# Create a dictionary and Document Term Matrix (DTM)
dictionary = corpora.Dictionary(tt_df['Clean_text'])
doc_term_matrix = [dictionary.doc2bow(doc) for doc in tt_df['Clean_text']]

# Generate the LDA model
lda_model = gensim.models.LdaModel(doc_term_matrix, num_topics=3, id2word=dictionary, passes=15)

# Print the topics and their associated words
for topic in lda_model.print_topics():
    print(topic)


# In[ ]:


#from gensim.models import Phrases
#from gensim.models import Word2Vec
#import ast
#import pandas as pd

# Assuming 'tt_df' is your DataFrame with the 'Clean_text' column
#corpus = tt_df['Clean_text'].tolist()  # Access the 'Clean_text' column as your corpus

# Learn and apply bigram phrases
#bigram = Phrases(corpus, min_count=5, threshold=5)
#corpus_with_bigrams = [bigram[sentence] for sentence in corpus]

# Learn and apply trigram phrases
#trigram = Phrases(corpus_with_bigrams, min_count=5, threshold=5)
#corpus_with_phrases = [trigram[sentence] for sentence in corpus_with_bigrams]

# Train Word2Vec model on the corpus with phrases
#model = Word2Vec(sentences=corpus_with_phrases, vector_size=100, window=5, min_count=7, sg=1)

# You can load the model later using model = Word2Vec.load("word2vec_model")


# In[17]:


from gensim.models import Phrases
from gensim.models import Word2Vec
import ast
import pandas as pd

# Assuming 'tt_df' is your DataFrame with the 'Clean_text' column
corpus = tt_df['Clean_text'].tolist()  # Access the 'Clean_text' column as your corpus

# Learn and apply bigram phrases
bigram = Phrases(corpus, min_count=20, threshold=6)
corpus_with_bigrams = [bigram[sentence] for sentence in corpus]

# Learn and apply trigram phrases
trigram = Phrases(corpus_with_bigrams, min_count=20, threshold=6)
corpus_with_phrases = [trigram[sentence] for sentence in corpus_with_bigrams]

# Validate n-grams in at least five input sentences
ngram_validation = {}

for sentence in corpus_with_phrases:
    unique_ngrams = set(sentence)
    
    for ngram in unique_ngrams:
        if ngram in ngram_validation:
            ngram_validation[ngram] += 1
        else:
            ngram_validation[ngram] = 1

# Filter n-grams that occur in at least five input sentences
min_sentence_count = 15
filtered_ngrams = [ngram for ngram, count in ngram_validation.items() if count >= min_sentence_count]

print("N-grams that occur in at least five input sentences:")
print(filtered_ngrams)

# Train Word2Vec model on the corpus with phrases
model = Word2Vec(sentences=corpus_with_phrases, vector_size=100, window=5, min_count=20, sg=1)


# In[18]:


corpus_with_phrases


# In[19]:


model.wv.most_similar("pandemic", topn=10)


# In[20]:


import pandas as pd


# Replace 'your_file.csv' with the path to your CSV file
csv_file = '/Users/maksymlesyk/Desktop/Anchor_words.csv'

# Read the CSV file and use the first row as headers
topic_anchors = pd.read_csv(csv_file, header=0)

# Now, 'df' is a DataFrame with the first row of the CSV file as column headers


# In[21]:


topic_anchors


# In[22]:


import pandas as pd

# Assuming 'topic_anchors' is your initial DataFrame and 'model' is your Word2Vec model
# Initialize a dictionary to store the new words for each topic
new_words_to_topics = {}

# Create a new DataFrame to store the new words
new_words_dataframe = pd.DataFrame()

# Iterate through each topic and its anchor words
for topic in topic_anchors.columns:
    anchor_words = topic_anchors[topic].dropna().tolist()
    
    # Initialize a list to store the new words for the current topic
    new_words = []

    # Calculate and collect new words from the model's vocabulary
    for anchor_word in anchor_words:
        if anchor_word in model.wv.key_to_index:
            similar = model.wv.most_similar(anchor_word, topn=20)  # Collect top 20 similar words per anchor word
            new_words.extend([(word, similarity) for word, similarity in similar])

    # Ensure uniqueness, select the top 20 new words with the greatest similarity, and filter out unwanted words
    unique_new_words = []
    seen_words = set()
    for word, similarity in new_words:
        if word not in seen_words and word not in anchor_words:
            unique_new_words.append((word, similarity))
            seen_words.add(word)
    
    # Sort the new words by similarity in descending order and select the top 20
    unique_new_words.sort(key=lambda x: x[1], reverse=True)
    unique_new_words = unique_new_words[:20]

    # Store the top new words for the current topic
    new_words_to_topics[topic] = [word for word, _ in unique_new_words]

    # Add the new words to the new DataFrame
    new_words_dataframe[topic] = [word for word, _ in unique_new_words]

# Print the 15 new words with the greatest similarity for each topic
for topic, new_words in new_words_to_topics.items():
    print(f"Top 20 new words for '{topic}':")
    for word in new_words:
        print(word)

# You now have the new words in the 'new_words_dataframe' DataFrame.
# You can save it to a CSV file if needed.
#new_words_dataframe.to_csv('new_words.csv', index=False)


# In[23]:


pd.concat([new_words_dataframe,topic_anchors],axis=0)


# In[24]:


import pandas as pd

# Assuming 'topic_anchors' is your initial DataFrame and 'model' is your Word2Vec model
# Get the vocabulary of the Word2Vec model
model_vocab = set(model.wv.key_to_index.keys())

# Create a new DataFrame with anchor words that exist in the model's vocabulary
filtered_topic_anchors = pd.DataFrame()

# Iterate through each topic and its anchor words
max_len = max(len(topic_anchors[topic]) for topic in topic_anchors.columns)

for topic in topic_anchors.columns:
    anchor_words = topic_anchors[topic].dropna().tolist()
    
    # Filter anchor words to keep only those in the model's vocabulary
    filtered_anchor_words = [word if word in model_vocab else None for word in anchor_words]

    # Ensure all columns have the same length by filling with None
    while len(filtered_anchor_words) < max_len:
        filtered_anchor_words.append(None)

    # Add the filtered anchor words to the new DataFrame
    filtered_topic_anchors[topic] = filtered_anchor_words

# Now 'filtered_topic_anchors' contains consistent-length columns, and missing cells are filled with None.


# In[25]:


filtered_topic_anchors


# In[26]:


final_anchor=pd.concat([new_words_dataframe,filtered_topic_anchors],axis=0)


# In[27]:


final_anchor


# In[ ]:


#import pandas as pd
#from sklearn.decomposition import TruncatedSVD
#import numpy as np

# Load your anchor words and text data into DataFrames

# Set the number of topics
#num_topics = len(final_anchor.columns)

# Initialize a dictionary to store the topic loadings
#topic_loadings = {}

# Iterate through each document in the 'NT_Clean_text' column
#for idx, doc in enumerate(tt_df['NT_Clean_text']):
#    print(f"Processing document {idx}...")
    
    # Initialize a dictionary to store topic loadings for the current document
#    doc_topic_loadings = {}
    
    # Iterate through each topic
#    for topic in final_anchor.columns:
        # Extract anchor words for the current topic
#        anchor_words = final_anchor[topic].dropna().tolist()
        
        # Create a document-term matrix for the current topic
#        topic_matrix = pd.DataFrame()
        
#        print(f"Topic: {topic}")
#        print(f"Anchor words: {anchor_words}")
        
        # Count word occurrences in the current document
#        for word in anchor_words:
#            count = doc.split().count(word)
#            if count > 0:
                # Add a small constant to the denominator to avoid division by zero
#                topic_matrix[word] = [count / (count + 1e-10)]
#            else:
                # If count is zero, set the value to 0
#                topic_matrix[word] = [0.0]
        
        # Normalize the row by dividing by the total word count
#        topic_matrix = topic_matrix.div(topic_matrix.sum(axis=1), axis=0)
#        topic_matrix = topic_matrix.fillna(0)
        #topic_matrix = topic_matrix.dropna()

#        print(f"Topic Matrix:")
#        print(topic_matrix)
        
        # Apply SVD to the normalized matrix
#        svd = TruncatedSVD(n_components=1)
#        topic_loading = svd.fit_transform(topic_matrix)
        
#        print(f"Topic Loading:")
#        print(topic_loading)
        
        # Ensure all loadings are positive
#        topic_loading = np.abs(topic_loading)
        
#        print(f"Positive Loading:")
#        print(topic_loading)
        
        # Store the topic loading for the current document and topic
#        doc_topic_loadings[topic] = topic_loading[0][0]
    
    # Store the topic loadings for the current document in the dictionary
#    topic_loadings[idx] = doc_topic_loadings

# Convert the topic loadings dictionary to a DataFrame
#topic_loadings_df = pd.DataFrame(topic_loadings).T

# The 'topic_loadings_df' DataFrame contains the topic loadings for each document.




# In[ ]:


tt_df


# In[31]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[ ]:


import pandas as pd
from sklearn.decomposition import TruncatedSVD
import numpy as np
import spacy

# Load your anchor words and text data into DataFrames

# Set the number of topics
num_topics = len(final_anchor.columns)

# Initialize a dictionary to store the topic loadings
topic_loadings = {}

# Load the spaCy model for tokenization
nlp = spacy.load("en_core_web_sm")

# Iterate through each document in the 'NT_Clean_text' column
for idx, doc in enumerate(tt_df['NT_Clean_text']):
    print(f"Processing document {idx}...")
    
    # Tokenize the cleaned text
    doc_tokens = [token.text for token in nlp(doc)]
    
    # Initialize a dictionary to store topic loadings for the current document
    doc_topic_loadings = {}
    
    # Iterate through each topic
    for topic in final_anchor.columns:
        # Extract anchor words for the current topic
        anchor_words = final_anchor[topic].dropna().tolist()
        
        # Create a document-term matrix for the current topic
        topic_matrix = pd.DataFrame()
        
        print(f"Topic: {topic}")
        print(f"Anchor words: {anchor_words}")
        
        # Count word occurrences in the current document
        for anchor_word in anchor_words:
            anchor_word_tokens = anchor_word.split('_')
            count = 0
            for n in range(1, len(anchor_word_tokens) + 1):
                for i in range(len(doc_tokens) - n + 1):
                    if doc_tokens[i:i+n] == anchor_word_tokens[:n]:
                        count += 1
            if count > 0:
                # Add a small constant to the denominator to avoid division by zero
                topic_matrix[anchor_word] = [count / (count + 1e-10)]
            else:
                # If count is zero, set the value to 0
                topic_matrix[anchor_word] = [0.0]
        
        # Normalize the row by dividing by the total word count
        topic_matrix = topic_matrix.div(topic_matrix.sum(axis=1), axis=0)
        topic_matrix = topic_matrix.fillna(0)

        print(f"Topic Matrix:")
        print(topic_matrix)
               
        # Apply SVD to the normalized matrix
        svd = TruncatedSVD(n_components=1)
        topic_loading = svd.fit_transform(topic_matrix)
 
        print(f"Topic Loading:")
        print(topic_loading)

        # Ensure all loadings are positive
        topic_loading = np.abs(topic_loading)
        
        print(f"Positive Loading:")
        print(topic_loading)
        
        # Store the topic loading for the current document and topic
        doc_topic_loadings[topic] = topic_loading[0][0]
    
    # Store the topic loadings for the current document in the dictionary
    topic_loadings[idx] = doc_topic_loadings

# Convert the topic loadings dictionary to a DataFrame
topic_loadings_df = pd.DataFrame(topic_loadings).T

# The 'topic_loadings_df' DataFrame contains the topic loadings for each document


# In[ ]:


topic_loadings_df


# In[ ]:


topic_loadings_df[topic_loadings_df['topic4']==0]


# In[ ]:




