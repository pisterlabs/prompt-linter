import pandas as pd
import string
import nltk
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from queue import PriorityQueue as pq
import editdistance
from nltk.corpus import words
import cohere
import streamlit as st 

adv_reads = pd.read_csv('google_books_1299.csv')
adv_reads.drop(columns=adv_reads.columns[0], axis=1, inplace=True)

adv_reads.drop_duplicates(keep='first', inplace=True, ignore_index=True)
adv_reads['lower_age'] = [16 for i in range(len(adv_reads))]
adv_reads['upper_age'] = [20 for i in range(len(adv_reads))]

child_books = pd.read_csv("children_books.csv")
child_books.drop_duplicates(keep='first', inplace=True, ignore_index=True)

# we want to replace reading age with appr. age lim
# new columns: upper_age, lower_age
upper_age = []
lower_age = []
for ind in child_books.index: 
    # print(child_books['Reading_age'][1][:2])
    if '-' in child_books['Reading_age'][ind]:
        lower, upper = child_books['Reading_age'][ind].split('-')
        upper_age.append(int(upper))
        lower_age.append(int(lower))
    else: # + 
        lower = int(child_books['Reading_age'][ind][:2].strip(string.punctuation))
        upper = int(lower+3) # manually set 
        upper_age.append(upper)
        lower_age.append(lower)

# add age columsn to child_books
child_books['lower_age'] = lower_age
child_books['upper_age'] = upper_age

child_stories = pd.read_csv('children_stories.Csv', encoding="ISO-8859-1")
child_stories.drop_duplicates(keep='first', inplace=True, ignore_index=True)
story_upper = []
story_lower = []
# Strip off age and split by -
for ind in child_stories.index:
    age = child_stories['cats'][ind][4:]
    if '-' in age:
        lower, upper = age.split('-')
        story_lower.append(int(lower))
        story_upper.append(int(upper))
    else: #+
        try: 
            lower = int(age.strip(string.punctuation))
            upper = lower + 3
            story_lower.append(int(lower))
            story_upper.append(int(upper))
        except: # in this case we will default to 2-9 years (default)
            story_lower.append(2)
            story_upper.append(9)
            
# Add 2 columns to child_stories
child_stories['lower_age'] = story_lower
child_stories['upper_age'] = story_upper

# we only need the title, upperage, lowerage, desc and author from each dataframe
df_1 = child_books.copy().drop(columns = ['Inerest_age', 'Reading_age'])
df_1['Type'] = 'Books'
df_1 = df_1.rename(columns = {'Title': 'title', 'Desc' : 'desc', 'Author':'author'})

df_2 = child_stories.copy().drop(columns = ['cats']) 
df_2['Type'] = 'Stories'
df_2 = df_2.rename(columns = {'names': 'title', 'Author':'author'})

df_combined = pd.concat([df_1, df_2])

# Rename columns of adv_reads 
try: 
    adv_reads.drop(columns=['rating', 'voters','price','currency','publisher','page_count','generes','ISBN','language','published_date'], inplace=True)
except:
    pass
adv_reads = adv_reads.rename(columns = {'description': 'desc'})
adv_reads['Type'] = ['Books' for i in range(len(adv_reads))]
adv_reads.fillna('', inplace=True)

# get rid of numbers
df_combined['title_formatted'] = df_combined['title'].str.replace('\d+', '')
df_combined['desc_formatted'] = df_combined['desc'].str.replace('\d+', '')
adv_reads['title_formatted'] = adv_reads['title'].str.replace('\d+','')
adv_reads['desc_formatted'] = adv_reads['desc'].str.replace('\d+','')

# make lowercase
df_combined['title_formatted'] = df_combined['title_formatted'].apply(str.lower)
df_combined['desc_formatted'] = df_combined['desc_formatted'].apply(str.lower)
adv_reads['title_formatted'] = adv_reads['title_formatted'].apply(str.lower)
adv_reads['desc_formatted'] = adv_reads['desc_formatted'].apply(str.lower)

# We can tokenize the words here to remove punctuation 
# Tokenize words 
nltk.download('punkt')
df_combined['title_desc'] = df_combined['title_formatted'].apply(word_tokenize) + df_combined['desc_formatted'].apply(word_tokenize)
adv_reads['title_desc'] = adv_reads['title_formatted'].apply(word_tokenize) + adv_reads['desc_formatted'].apply(word_tokenize)

# Remove stopwords
nltk.download('stopwords')
def remove_stopwords(input1):
    words = []
    for word in input1:
        if word not in stopwords.words('english'):
            words.append(word)
    return words

df_combined['title_desc'] =  df_combined['title_desc'].apply(remove_stopwords)
adv_reads['title_desc'] =  adv_reads['title_desc'].apply(remove_stopwords)

# Lemmatize 
nltk.download('wordnet')
nltk.download('omw-1.4')
lem = WordNetLemmatizer()
def lemma_wordnet(input):
    return [lem.lemmatize(w) for w in input]
df_combined['title_desc'] = df_combined['title_desc'].apply(lemma_wordnet)
adv_reads['title_desc'] = adv_reads['title_desc'].apply(lemma_wordnet)

# translate list to string
def join_text(input):
    combined = ' '.join(input)
    return combined
df_combined['title_desc_filtered']=df_combined['title_desc'].apply(join_text)
adv_reads['title_desc_filtered']=adv_reads['title_desc'].apply(join_text)

# Remove punctuation for clarity 
df_combined['title_desc_filtered'] = df_combined['title_desc_filtered'].str.replace(r'[^\w\s]+', '')

st.write(
    '''
    # Summer Reading made fun! 
    '''
)
st.write(
    '''
    ## Enjoy reading a book based on your current interests! 
    '''
)
user_age = st.number_input("Select your age", 15)

user_age = int(user_age)
if int(user_age) >= 16: 
    df_16_plus = pd.concat([df_combined, adv_reads])
    df_filter = df_16_plus[(df_16_plus['lower_age'] <= user_age) & (df_16_plus['upper_age'] >= user_age)]
else: 
    df_filter = df_combined[(df_combined['lower_age']<=user_age) & (df_combined['upper_age']>=user_age) ]

vectorizer = TfidfVectorizer(max_features=1800, lowercase=True, stop_words='english', ngram_range=(1,1)) #unigrams and bigrams
tf_idf_output = vectorizer.fit_transform(df_filter['title_desc_filtered'])
vocab = np.array(vectorizer.get_feature_names())

num_topics=25 # can be changed for fine-tuning

# Perform LDA 
num_topics = int(num_topics) # can be changed 
lda = LatentDirichletAllocation(n_components=num_topics, random_state=1).fit(tf_idf_output)

num_top_words=15

num_top_words = int(num_top_words) # can be changed. Can also be generated from Cohere
  
topics_set = []

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        topics_set.append(set([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(lda, vocab, num_top_words)

user_phrase = st.text_input("Enter some words that describe what you want to read", "small town murder that results in an investigation case reported from the victim")
# user_phrase="small town murder that results in an investigation case reported from the victim"

# word correct
nltk.download('words')
words_corpus = words.words()

user_input_list = user_phrase.split()
user_input_list = [elem.strip(string.punctuation).lower() for elem in user_input_list]

def run_autocorrect(input_word): 
    closest_words = pq()
    for word in words_corpus: 
        distance = editdistance.eval(word, input_word)
        closest_words.put((distance, word))
    return closest_words.get()

new_phrase_list = []
for elem in user_input_list: 
    if elem not in words_corpus: 
        correct_word = run_autocorrect(elem)
        new_phrase_list.append(correct_word[1])
    else: 
        new_phrase_list.append(elem)

user_phrase = " ".join(new_phrase_list)

# There is no point of removing stopwords here because we are seeing how much similarity there is regardless. 
# I am thinking of adding spell-correction here? 

# user_phrase = set(user_phrase.lower().split())
# we will preprocess the user phrase just like how we preprocessed the descriptions. Removing stopwords won't matter
# since we are looking at frequency 

user_phrase = word_tokenize(user_phrase)
user_phrase = [word if word not in string.punctuation else '' for word in user_phrase]
 #user_phrase.replace(string.punctuation, '')
user_phrase = set([lem.lemmatize(w).lower() for w in user_phrase])
# print(user_phrase)

best_topic = topics_set[0] #initialization 
best_intersection = -1
for group in topics_set: 
    # print(group, user_phrase, len(user_phrase.intersection(group)))
    print(group)
    if len(user_phrase.intersection(group)) > best_intersection: 
        best_topic = group
        best_intersection = len(user_phrase.intersection(group))

# print(best_topic, best_intersection) # returns best list of topics and length of the intersection 

# Now we need to match the book descriptions to the topic that was chosen 
books = pq() #min heap 
for index, row in df_filter.iterrows():
    # print(row['title_desc_filtered'])
    intersect_len = len(set(row['title_desc_filtered'].split()).intersection(best_topic))
    # print(set(row['title_desc_filtered'].split()))
    # if pd.isna(row['author']): 
    #     books.put(books.put((-1*intersect_len, row['title'].strip(string.punctuation), "None", row['Type'].strip(string.punctuation))))
    # else: 
    books.put((-1*intersect_len, row['title'], row['author'], row['Type']))

num_recs=5

books_title = []
books_author = []
books_type = []
while len(set(books_title)) < 5: 
    book = books.get()
    # print(book[0])
    if book[1] not in books_title: 
        books_title.append(book[1])
        books_author.append(book[2])
        books_type.append(book[3])

book_recs_df = pd.DataFrame()
book_recs_df['Title'] = books_title
book_recs_df['Author'] = books_author
book_recs_df['Type'] = books_type

st.write(
    '''
    # Your recommendations are: 
    '''
)

st.table(book_recs_df)

st.write("# Didn't find a book that suits your taste? How about writing one yourself?")
st.write("## Generate an idea here!")
theme = st.text_input("Enter some keywords that describe what you want to write about", "machine learning is complex, yet fascinating")

#'crime, murder, new york city, suspect, man, FBI, detective'

import cohere
co = cohere.Client('KTQmgyEWSk81jeyS98xCMB1iRuWjZ5KDzSkrdw0b')
prediction = co.generate(
  model='large',
  prompt='--\nProduct: Book \nKeywords: ' + str(theme) + str('Exciting Book Description:'),
  max_tokens=50,
  temperature=0.8,
  k=0,
  p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop_sequences=["--"],
  return_likelihoods='NONE')
print('Idea: {}...'.format(prediction.generations[0].text))
st.write("Sample Storyline: ",str(prediction.generations[0].text) + str("..."))

