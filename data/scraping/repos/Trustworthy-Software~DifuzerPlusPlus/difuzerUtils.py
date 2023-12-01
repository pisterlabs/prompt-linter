# IMPORT
from nltk                   import pos_tag
from nltk.corpus            import stopwords
from nltk.corpus            import wordnet
from nltk.stem              import SnowballStemmer
from nltk.stem.wordnet      import WordNetLemmatizer
from nltk.tokenize          import word_tokenize
import nltk, re
import subprocess
import requests
import numpy as np
import os
import re
import string
import enchant

# Difuzer++
# 
# Copyright (C) 2023 Marco Alecci
# University of Luxembourg - Interdisciplinary Centre for
# Security Reliability and Trust (SnT) - TruX - All rights reserved
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Lesser Public License for more details.
# 
# You should have received a copy of the GNU General Lesser Public
# License along with this program.  If not, see <http://www.gnu.org/licenses/lgpl-2.1.html>.

################## API KEYS ########################
from   dotenv import load_dotenv
import os,sys
# Load API KEYS from the .env file in the current directory
CONFIG_PATH = "./config.env"
if not os.path.exists(CONFIG_PATH):
    print(f"⚠️ Error: File not found at path '{CONFIG_PATH}'.\n- Make sure the config.env file exists.\n- Ensure the CONFIG_PATH is correctly set.")
    sys.exit(1)
else:
    load_dotenv(CONFIG_PATH)
ANDROZOO_API_KEY = os.getenv('ANDROZOO_API_KEY')
OPENAI_API_KEY   = os.getenv('OPENAI_API_KEY')
ANDROID_PLATFORM_PATH = os.environ.get('ANDROID_PLATFORM_PATH')
APK_PATH              = os.environ.get('APK_PATH')
#######################################################

# extractFeatures()
# 1) Download APK from AndroZoo
# 2) Launch Difuzer with all details about possible logic bombs (filtering applied)
# 3) Combine the features using predefined delimitators and return them
def extractFeatures(sha256):

    # Download apk from Androzoo
    apkUrl = "https://androzoo.uni.lu/api/download?apikey={}&sha256={}".format(ANDROZOO_API_KEY, sha256)
    req = requests.get(apkUrl, allow_redirects=True)
    open(APK_PATH+'{}.apk'.format(sha256), "wb").write(req.content)

    # Get the features using Difuzer
    command = 'java -jar ./Difuzer-0.1-jar-with-dependencies.jar -a {}{}.apk -p {}'.format(APK_PATH, sha256, ANDROID_PLATFORM_PATH)
    print("EXECUTING: {}\n".format(command))
    
    # Output from Difuzer
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Remove apk file
    os.remove(APK_PATH + '{}.apk'.format(sha256))

    # Reorganize features using predefined delimitators
    triggersFeaturesList = output.decode("utf-8").split("@@@\n")[:-1]

    # If empty list
    if len(triggersFeaturesList) == 0:
        return np.nan
        
    for i in range(0,len(triggersFeaturesList)):
        triggersFeaturesList[i] = triggersFeaturesList[i].replace("\n",";")

    if triggersFeaturesList[0] != "":
        for i in range(0, len(triggersFeaturesList)):
            triggersFeaturesList[i] = triggersFeaturesList[i].split(";")
            triggersFeaturesList[i][3] = triggersFeaturesList[i][3].split("$$$")[:-1]

    # Return
    if triggersFeaturesList is not np.nan:
        return triggersFeaturesList
    else:
        return np.nan

# Print a trigger
def printTrigger(trigger):
    fv          = trigger[0]
    method      = trigger[1]
    condition   = trigger[2]
    sources     = trigger[3]

    print("\n⚠️ 💣 - Possible Logic Bomb")
    print("FV       : {}".format(fv))
    print("Method   : {}".format(method))
    print("Condition: {}".format(condition))
    print("Sources  :")
    for s in sources:
        print("     - {}".format(s))

    return

# Get the fields of a trigger
def getTriggerMethodAndSources(trigger):
    fv          = trigger[0]
    method      = trigger[1]
    condition   = trigger[2]
    sources     = trigger[3]

    return method, sources

# Get the ID of the topic assigned by the LDA Model
def getLdaID(vectorizer, ldaModel ,description):
    
    # Needed for NLP
    st              = nltk.stem.snowball.EnglishStemmer()
    english_vocab   = set(w.lower() for w in nltk.corpus.words.words())
    stopwords       = nltk.corpus.stopwords.words('english')
    corpus          = []

    string = description
    string = re.sub(r'\W',' ',string)
    string = re.sub(r'\d','',string)
    tokens = nltk.word_tokenize(string)
    words  = [st.stem(w) for w in tokens if len(w)>=3 and w.lower() not in stopwords and w.lower() in english_vocab]          
    descriptionProcessed = ' '.join(words)       
    corpus.append(descriptionProcessed)

    # Retrieve the Topic ID 
    tf_array  = vectorizer.transform(corpus)
    doc_topic = ldaModel.transform(tf_array)
    lda_id    = doc_topic[0].argmax()

    return lda_id

# Get the ID of the cluster assigned by the KMeans Model
def getKmeansID(vectorizer, kmeansModel ,description):
    
    # Needed for NLP
    st              = nltk.stem.snowball.EnglishStemmer()
    english_vocab   = set(w.lower() for w in nltk.corpus.words.words())
    stopwords       = nltk.corpus.stopwords.words('english')
    corpus          = []

    string = description
    string = re.sub(r'\W',' ',string)
    string = re.sub(r'\d','',string)
    tokens = nltk.word_tokenize(string)
    words  = [st.stem(w) for w in tokens if len(w)>=3 and w.lower() not in stopwords and w.lower() in english_vocab]          
    descriptionProcessed = ' '.join(words)       
    corpus.append(descriptionProcessed)

    # Retrieve the kmeans ID 
    tf_array  = vectorizer.transform(corpus)
    kmeans_id = kmeansModel.predict(tf_array)

    return int(kmeans_id)

import openai
openai.api_key = OPENAI_API_KEY

def getGptEmbedding(text):
   # Model to be used - (Determine the price)
   model="text-embedding-ada-002"

   # Remove new line chars
   text = text.replace("\n", " ")
   
   # Return Embedding
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


def getGcataID(kmeansModel ,description):
    description = preprocessDescriptionGcata(description)

    embedding = getGptEmbedding(description)
    embedding = np.array(embedding).reshape(1, -1)

    clusterID = kmeansModel.predict(embedding)

    return int(clusterID)


###################### PREPROCESSING G-CatA ################################s
# Remove all html tags from text
def remove_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Remove all words that are not in the US+GB dictionary
def remove_non_english(text):
    d_us = enchant.Dict("en_US")
    d_gb = enchant.Dict("en_GB")
    # build custom dict
    alt_dicts = []
    for d_file in os.listdir('./0_Data/no-filter-dict'):
        with open(os.path.join('./0_Data/no-filter-dict', d_file), 'r') as ff:
            list_terms = ff.read().lower().split()
            alt_dicts.extend(list_terms)
    new_text = ''
    for t in text.split():
        # if text is eith in US or GB english dict keep it
        if d_us.check(t) or d_gb.check(t):
            new_text = new_text + t + " "
            continue
        if t in alt_dicts:
            new_text = new_text + t + " "
            continue
    return new_text

# Replace non ascii chars with spaces
def remove_non_ascii(text):
    printable = set(string.printable)
    return filter(lambda x: x in printable, text)

# Replace punctuation chars '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' with spaces
def remove_punctuation(text):
    return ''.join(map(lambda c: ' ' if c in string.punctuation else c, text))

# Remove words with no meaning or irrelevant for searching
def remove_stopwords(text):

    return ' '.join([s for s in text.split() if s not in
                     stopwords.words('english')])

# Extracts the root of every word 
def apply_stemming(text):
    stemmer = SnowballStemmer("english")
    tokens = word_tokenize(text)
    stemmed_tokens = map(stemmer.stem, tokens)
    return ' '.join(stemmed_tokens)

# Part of Speech Detection
def wordnet_pos_code(tag):
    if tag.startswith('NN'):
        return wordnet.NOUN
    elif tag.startswith('VB'):
        return wordnet.VERB
    elif tag.startswith('JJ'):
        return wordnet.ADJ
    elif tag.startswith('RB'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default value

# Lemmatization
def apply_lemmatization(text):
    lmtzr = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos_tokens = pos_tag(tokens)
    lemm_tokens = []
    for token, pos in pos_tokens:
        w = lmtzr.lemmatize(token, wordnet_pos_code(pos))
        lemm_tokens.append(w)
    return ' '.join(lemm_tokens)

# Remove numbers from text
def remove_numbers(text):
    
    clean = re.compile('[0-9]')
    return re.sub(clean, '', text)

# Text to lowercase, for stopwords and stuff
def do_lowercase(text):  
    return text.lower()

# Preprocess Descriptions
def preprocessDescriptionGcata(description):

    # Apply all the steps
    description = remove_html(description)
    description = remove_non_ascii(description)
    description = remove_punctuation(description)
    description = do_lowercase(description)
    description = remove_numbers(description)
    description = remove_stopwords(description)
    description = remove_non_english(description)
    description = apply_lemmatization(description)

    return description
