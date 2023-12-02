#%%

import mysql.connector as sql
from datetime import date
import pandas as pd
import numpy
import fasttext
import re

def get_site_tag_data():
    #### Initlialize mysql connection
    db_connection = sql.connect( host='localhost', port= 3306,
                                database='dp3_database', user='root', password=1234) 
                                
    cursor = db_connection.cursor()

    cursor.execute(
        ''
    )
    
    table_rows = cursor.fetchall()
    

#%%
#%%
from analytics_utility import *
import mysql.connector as sql
from datetime import date
import pandas as pd
import numpy
import fasttext
import re

#### Initlialize mysql connection
db_connection = sql.connect( host='localhost', port= '3306',
                            database='dp3_database', user='root', password='1234') 
                      
cursor = db_connection.cursor()

cursor.execute(
    '''
        SELECT parser_group_url_tag_info.id, tag_id as tagId, parent_id as parentId, url_group_id as groupId, depth, tag, text, xpath, select_tag as selectTag,
            select_child_tags as selectChildTags, tag_data_type_id as tagDataTypeId, site_dict_tag_data_type.code
    FROM parser_group_url_tag_info
    LEFT JOIN site_dict_tag_data_type ON
        parser_group_url_tag_info.tag_data_type_id = site_dict_tag_data_type.id
    WHERE
        LENGTH(text) > 1
    LIMIT 25000
''')

table_rows = cursor.fetchall()

columns = [d[0] for d in cursor.description]
df = pd.DataFrame(table_rows, columns=columns)

df = transform_text(df)
df_en, df_kz, df_ru = identify_lanuage_split_train_test(df)

#%%
import snowballstemmer
import nltk
# nltk.download('stopwords')
language = 'russian'
stopwords = nltk.corpus.stopwords.words(language)
stemmer = snowballstemmer.stemmer(language);

# frequencies = df_ru['text'].value_counts().reset_index()

words = df_ru['text'].str.split(expand=True).stack()
stemmed_words = stemmer.stemWords(words)
stemmed_words = pd.Series(stemmed_words)
# stemmed_words.value_counts().reset_index()

# Filter stopwords
filtered_words = stemmed_words[~stemmed_words.isin(stopwords)]
# Filter numbers out
filtered_words = filtered_words[pd.to_numeric(filtered_words, errors='coerce').isna()]

frequencies = filtered_words.value_counts().reset_index()

sorted_frequencies = frequencies.sort_values(by='count', ascending=True)

result = sorted_frequencies.values.tolist()

result
#%%


#%%
import pandas as pd
from nltk import ngrams
from nltk.tokenize import word_tokenize

# Sample text data
data = pd.Series(["This is a sample sentence.", "Another example sentence."])

# Tokenize and generate n-grams
n = 2  # Define the size of n-grams
ngram_list = []

for text in data:
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase the text
    grams = ngrams(tokens, n)  # Generate n-grams
    ngram_list.extend([' '.join(gram) for gram in grams])  # Convert n-grams to strings and add to the list

# Convert n-grams to a new DataFrame
ngram_df = pd.DataFrame({'ngram': ngram_list})

# Display the resulting DataFrame
print(ngram_df)
#%%

import pandas as pd
from nltk import ngrams
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Sample text data
data = pd.Series(["This is a sample sentence.", "Another example sentence."])

# Tokenize and generate n-grams
n = 2  # Define the size of n-grams
ngram_list = []

for text in data:
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase the text
    grams = ngrams(tokens, n)  # Generate n-grams
    ngram_list.extend([' '.join(gram) for gram in grams])  # Convert n-grams to strings and add to the list

# Train Word2Vec embeddings
embedding_size = 100  # Define the dimensionality of the embeddings
window_size = 5  # Define the size of the context window
min_count = 1  # Minimum frequency threshold for n-grams

ngram_sentences = [ngram.split() for ngram in ngram_list]  # Prepare n-grams as sentences for Word2Vec
model = Word2Vec(ngram_sentences, vector_size=embedding_size, window=window_size, min_count=min_count)

# Check if the target n-gram is present in the vocabulary
target_ngram = "sample sentence"
if target_ngram in model.wv.key_to_index:
    embedding_vector = model.wv[target_ngram]
    print(f"Embedding vector for '{target_ngram}':")
    print(embedding_vector)
else:
    print(f"'{target_ngram}' is not present in the vocabulary.")
#%%
import pandas as pd
from nltk import ngrams
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Sample text data
data = pd.Series(["This is a sample sentence.", "Another example sentence."])

# Tokenize and generate n-grams
n = 2  # Define the size of n-grams
ngram_list = []

for text in data:
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase the text
    grams = ngrams(tokens, n)  # Generate n-grams
    ngram_list.extend([' '.join(gram) for gram in grams])  # Convert n-grams to strings and add to the list

# Preprocess n-grams to treat phrases as single tokens
phrases = ["sample sentence", "another example"]
preprocessed_ngram_list = [phrase.replace(" ", "_") if phrase in ngram_list else phrase for phrase in ngram_list]

# Train Word2Vec embeddings
embedding_size = 100  # Define the dimensionality of the embeddings
window_size = 5  # Define the size of the context window
min_count = 1  # Minimum frequency threshold for n-grams

ngram_sentences = [ngram.split() for ngram in preprocessed_ngram_list]  # Prepare preprocessed n-grams as sentences for Word2Vec
model = Word2Vec(ngram_sentences, vector_size=embedding_size, window=window_size, min_count=min_count)

# Get embedding vector for a specific phrase
target_phrase = "sample"
if target_phrase in model.wv.key_to_index:
    embedding_vector = model.wv[target_phrase]
    print(f"Embedding vector for '{target_phrase}':")
    print(embedding_vector)
else:
    print(f"'{target_phrase}' is not present in the vocabulary.")

#%%

import os
os.environ['OPENAI_API_KEY'] = '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'
import guidance
#%%

guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo")

#  'social sciences in general', 'philosophy', 'history. historical sciences', 'sociology', 'demography', 'economy and economic sciences', 'state and law. legal sciences', 'politics and political sciences', 'instruction', 'culture. culturology', 'public education. pedagogy', 'psychology', 'linguistics', 'literature. literary studies. folklore', 'art. art', 'mass communication. journalism. mass media', 'computer science', 'religion. atheism', 'comprehensive country and regional studies', 'complex problems of social sciences', 'math', 'cybernetics', 'physics', 'mechanics', 'chemistry', 'biology', 'geodesy. cartography', 'geophysics', 'geology', 'geography', 'astronomy', 'general and complex problems of the natural and exact sciences', 'energy', 'electrical', 'electronics. radio engineering', 'communication', 'automatic. computer engineering', 'mining', 'metallurgy', 'engineering', 'nuclear engineering', 'instrument making', 'polygraphy. reprography. photo cinema', 'chemical technology. chemical industry', 'biotechnology', 'light industry', 'food industry', 'forestry and wood processing', 'construction. architecture', 'agriculture and forestry', 'fishing. aquaculture', 'water management', 'domestic trade. tourist and excursion service', 'foreign trade', 'transport', 'housing and utilities. home economy. household service', 'medicine and health care', 'physical culture and sport', 'military', 'other sectors of the economy', 'general and complex problems of technical and applied sciences and branches of the national economy', 'organization and management', 'statistics', 'standardization', 'patent case. invention. innovation', 'safety', 'environment protection. human ecology', 'space research', 'metrology'
#  social sciences in general, philosophy, history. historical sciences, sociology, demography, economy and economic sciences, state and law. legal sciences, politics and political sciences, instruction, culture. culturology, public education. pedagogy, psychology, linguistics, literature. literary studies. folklore, art. art, mass communication. journalism. mass media, computer science, religion. atheism, comprehensive country and regional studies, complex problems of social sciences, math, cybernetics, physics, mechanics, chemistry, biology, geodesy. cartography, geophysics, geology, geography, astronomy, general and complex problems of the natural and exact sciences, energy, electrical, electronics. radio engineering, communication, automatic. computer engineering, mining, metallurgy, engineering, nuclear engineering, instrument making, polygraphy. reprography. photo cinema, chemical technology. chemical industry, biotechnology, light industry, food industry, forestry and wood processing, construction. architecture, agriculture and forestry, fishing. aquaculture, water management, domestic trade. tourist and excursion service, foreign trade, transport, housing and utilities. home economy. household service, medicine and health care, physical culture and sport, military, other sectors of the economy, general and complex problems of technical and applied sciences and branches of the national economy, organization and management, statistics, standardization, patent case. invention. innovation, safety, environment protection. human ecology, space research, metrology,
 
prompt = guidance(
'''

    {{#user~}}
    {{q1}}
    {{~/user}}

    {{#assistant~}}
    {{gen 'response' temperature=0.1 max_tokens=13 stop="<|im_end|>"}}
    {{~/assistant}}

''')
prompt = prompt(q1 = '''
            This is the list of topics:
            'social sciences in general', 'philosophy', 'history. historical sciences', 'sociology', 'demography', 'economy and economic sciences', 'state and law. legal sciences','politics and political sciences', 'instruction', 'culture. culturology', 'public education. pedagogy', 'psychology', 'linguistics', 'literature. literary studies. folklore', 'art. art', 'mass communication. journalism. mass media', 'computer science', 'religion. atheism', 'comprehensive country and regional studies', 'complex problems of social sciences', 'math', 'cybernetics', 'physics', 'mechanics', 'chemistry', 'biology', 'geodesy. cartography', 'geophysics', 'geology', 'geography', 'astronomy', 'general and complex problems of the natural and exact sciences', 'energy', 'electrical', 'electronics. radio engineering', 'communication', 'automatic. computer engineering', 'mining', 'metallurgy', 'engineering', 'nuclear engineering', 'instrument making', 'polygraphy. reprography. photo cinema', 'chemical technology. chemical industry', 'biotechnology', 'light industry', 'food industry', 'forestry and wood processing', 'construction. architecture', 'agriculture and forestry', 'fishing. aquaculture', 'water management', 'domestic trade. tourist and excursion service', 'foreign trade', 'transport', 'housing and utilities. home economy. household service', 'medicine and health care', 'physical culture and sport', 'military', 'other sectors of the economy', 'general and complex problems of technical and applied sciences and branches of the national economy', 'organization and management', 'statistics', 'standardization', 'patent case. invention. innovation', 'safety', 'environment protection. human ecology', 'space research', 'metrology'
            "Geothermal deposits are the most promising source of energ" assign each topics
''',)
prompt
#%%


llm = guidance.llms.OpenAI('text-davinci-001')
program = guidance("""My favorite flavor is{{gen 'flavor' max_tokens=1 stop="."}}""", llm=llm)
program()

#%%

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Define the list of topics
topics = [
    'social sciences in general', 'philosophy', 'history. historical sciences',
    'sociology', 'demography', 'economy and economic sciences',
    'state and law. legal sciences', 'politics and political sciences',
    'instruction', 'culture. culturology', 'public education. pedagogy',
    'psychology', 'linguistics', 'literature. literary studies. folklore',
    'art. art', 'mass communication. journalism. mass media',
    'computer science', 'religion. atheism', 'comprehensive country and regional studies',
    'complex problems of social sciences', 'math', 'cybernetics', 'physics',
    'mechanics', 'chemistry', 'biology', 'geodesy. cartography', 'geophysics',
    'geology', 'geography', 'astronomy',
    'general and complex problems of the natural and exact sciences',
    'energy', 'electrical', 'electronics. radio engineering',
    'communication', 'automatic. computer engineering',
    'mining', 'metallurgy', 'engineering', 'nuclear engineering',
    'instrument making', 'polygraphy. reprography. photo cinema',
    'chemical technology. chemical industry', 'biotechnology',
    'light industry', 'food industry', 'forestry and wood processing',
    'construction. architecture', 'agriculture and forestry',
    'fishing. aquaculture', 'water management',
    'domestic trade. tourist and excursion service', 'foreign trade',
    'transport', 'housing and utilities. home economy. household service',
    'medicine and health care', 'physical culture and sport',
    'military', 'other sectors of the economy',
    'general and complex problems of technical and applied sciences and branches of the national economy',
    'organization and management', 'statistics', 'standardization',
    'patent case. invention. innovation', 'safety',
    'environment protection. human ecology', 'space research', 'metrology'
]

# Assign categories to each topic
category_mapping = {
    'Social Sciences': [
        'social sciences in general', 'philosophy', 'history. historical sciences',
        'sociology', 'demography', 'economy and economic sciences',
        'state and law. legal sciences', 'politics and political sciences',
        'instruction', 'culture. culturology', 'public education. pedagogy',
        'psychology', 'linguistics', 'literature. literary studies. folklore',
        'art. art', 'mass communication. journalism. mass media',
        'computer science', 'religion. atheism', 'comprehensive country and regional studies',
        'complex problems of social sciences'
    ],
    'Natural and Exact Sciences': [
        'math', 'cybernetics', 'physics',
        'mechanics', 'chemistry', 'biology', 'geodesy. cartography', 'geophysics',
        'geology', 'geography', 'astronomy',
        'general and complex problems of the natural and exact sciences'
    ],
    'Engineering and Technology': [
        'energy', 'electrical', 'electronics. radio engineering',
        'communication', 'automatic. computer engineering',
        'mining', 'metallurgy', 'engineering', 'nuclear engineering',
        'instrument making', 'polygraphy. reprography. photo cinema',
        'chemical technology. chemical industry', 'biotechnology',
        'light industry', 'food industry', 'forestry and wood processing',
        'construction. architecture', 'agriculture and forestry',
        'fishing. aquaculture', 'water management',
        'domestic trade. tourist and excursion service', 'foreign trade',
        'transport', 'housing and utilities. home economy. household service'
    ],
    'Other Fields': [
        'medicine and health care', 'physical culture and sport',
        'military', 'other sectors of the economy',
        'general and complex problems of technical and applied sciences and branches of the national economy',
        'organization and management', 'statistics', 'standardization',
        'patent case. invention. innovation', 'safety',
        'environment protection. human ecology', 'space research', 'metrology'
    ]
}

dataset = []
labels = []
for topic in topics:
    for category, topics_in_category in category_mapping.items():
        if topic in topics_in_category:
            dataset.append([topic])
            labels.append(category)
            break

# One-hot encode the features
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
onehot_encoder = OneHotEncoder(sparse=False)
dataset_encoded = onehot_encoder.fit_transform(dataset)

# Create and train the AdaBoost classifier
classifier = AdaBoostClassifier()
classifier.fit(dataset_encoded, labels_encoded)
# dataset
# Predict the category for a new example
new_example = [['Other Fields']]

new_example_encoded = onehot_encoder.transform(new_example)
classifier.predict(new_example_encoded)
# predicted_category = label_encoder.inverse_transform()

# print("Predicted category:", predicted_category[0])

#%%
import openai

# Set up OpenAI API credentials
openai.api_key = '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'

# Define the example text
example_text = "Geothermal deposits are the most promising source of energy."

# Define the list of topics
topics = [
    'social sciences in general', 'philosophy', 'history. historical sciences',
    'sociology', 'demography', 'economy and economic sciences',
    'state and law. legal sciences', 'politics and political sciences',
    'instruction', 'culture. culturology', 'public education. pedagogy',
    'psychology', 'linguistics', 'literature. literary studies. folklore',
    'art. art', 'mass communication. journalism. mass media',
    'computer science', 'religion. atheism', 'comprehensive country and regional studies',
    'complex problems of social sciences', 'math', 'cybernetics', 'physics',
    'mechanics', 'chemistry', 'biology', 'geodesy. cartography', 'geophysics',
    'geology', 'geography', 'astronomy',
    'general and complex problems of the natural and exact sciences',
    'energy', 'electrical', 'electronics. radio engineering',
    'communication', 'automatic. computer engineering',
    'mining', 'metallurgy', 'engineering', 'nuclear engineering',
    'instrument making', 'polygraphy. reprography. photo cinema',
    'chemical technology. chemical industry', 'biotechnology',
    'light industry', 'food industry', 'forestry and wood processing',
    'construction. architecture', 'agriculture and forestry',
    'fishing. aquaculture', 'water management',
    'domestic trade. tourist and excursion service', 'foreign trade',
    'transport', 'housing and utilities. home economy. household service',
    'medicine and health care', 'physical culture and sport',
    'military', 'other sectors of the economy',
    'general and complex problems of technical and applied sciences and branches of the national economy',
    'organization and management', 'statistics', 'standardization',
    'patent case. invention. innovation', 'safety',
    'environment protection. human ecology', 'space research', 'metrology'
]

# Define the prompt for GPT-3
prompt = "Assign a topic to the following statement: {}Topic:".format(example_text)

# Generate a response from GPT-3
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=1,
    n=1,
    stop=None,
    temperature=0.3
)

# Get the generated topic from the response
generated_topic = response.choices[0].text.strip()

# Find the closest matching topic from the list of topics
closest_topic = min(topics, key=lambda x: len(set(x.split()) & set(generated_topic.split())))

print("Generated Topic:", generated_topic)
print("Closest Matched Topic:", closest_topic)