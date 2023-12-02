import sys, os
sys.path.append('../')
import streamlit as st
import pandas as pd
import json
import config
import spacy
import string
import gensim
import nltk
import es_core_news_sm
import re
import unidecode
import pickle
import time
import altair as alt
import pyLDAvis.gensim

from pandas.io.json import json_normalize
from spacy.lang.es import Spanish
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from coherence_computation import *


#initialize parser and stop_words
nltk.download("stopwords")
nlp = es_core_news_sm.load()
parser = Spanish()
es_stop = set(nltk.corpus.stopwords.words("spanish"))

CHANNEL_ID = "UCoGBPBXyq28cE4g2TaB6lRQ"  #DamianKucOK
VIDEO_ID = "RHlFYRonmj4"  #Shoko Asahara
MAX_RESULTS = 10
YOUTUBE_BASE_API_URL = "https://www.googleapis.com/youtube/v3/"

# comments_data.json = "https://www.googleapis.com/youtube/v3/commentThreads?part=snippet,replies&allThreadsRelatedToChannelId=UCoGBPBXyq28cE4g2TaB6lRQ&key=API_KEY&maxResults=100&order=time"
# comments de canal (canal y videos) = "https://www.googleapis.com/youtube/v3/commentThreads?part=snippet,replies&allThreadsRelatedToChannelId={CHANNEL_ID}&key={API_KEY}&maxResults=100&order=time"

# endpoints youtube api https://stackoverflow.com/questions/18953499/youtube-api-to-fetch-all-videos-on-a-channel

try:
    from urllib.request import Request, urlopen  # Python 3
except ImportError:
    from urllib2 import Request, urlopen  # Python 2

#@st.cache
#TODO not being used for now
def get_data_into_file(filename):
  url = YOUTUBE_BASE_API_URL+"commentThreads?key={}&textFormat=plainText&part=snippet&videoId={}&maxResults={}".format(config.API_KEY, VIDEO_ID, MAX_RESULTS)
  response = urlopen(url)
  json_data = response.read().decode('utf-8', 'replace')
  import pdb; pdb.set_trace()
  data = json.loads(json_data)['items']
  json_data = json_normalize(data)

  with open(filename, 'w') as outfile:
      json.dump(json_data, outfile)  

def load_data_from(filename):
  with open(filename) as json_file:
      data = json.load(json_file)['items']
      comments = []
      for item in data:
        comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
        comments.append(comment)
      
      return pd.Series(comments)


st.title("Comentarios de videos")
#get_data_into_file("data.json")
#comments_video = load_data_from("data/video_data.json")
#st.write(comments_video)
placeholder = st.empty()
placeholder.text("Cargando comentarios")
last_1000_comments = pd.Series()
for i in range(10):
  placeholder.text("Cargando {}%".format((i+1)*10))
  last_1000_comments = last_1000_comments.append(load_data_from("../data/comments_data_{}.json".format(i)), ignore_index=True)

st.success("{} comentarios agregados".format(len(last_1000_comments)))
placeholder.empty()

def deEmojify(text):
  regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", flags = re.UNICODE)
  return regrex_pattern.sub(r'',text)

def clean_text(text):
  text = text.strip()
  text = text.lower()
  text = unidecode.unidecode(text) #accents
  text = deEmojify(text) #emojis
  laugh_pattern = re.compile(pattern = "^(j[aeiou])+") #laughs
  text = laugh_pattern.sub(r'', text)
  return text

def tokenize(comment):
  lda_tokens = []
  tokens = parser(comment)
  for token in tokens:
    if token.is_space or token.is_punct:
      continue
    elif token.like_url:
      lda_tokens.append("URL")
    else:
      lda_tokens.append(token.lower_)
  return lda_tokens

# TODO Terms that dont belong to the business case
#terms_dictionary = ["buenas noches", "buenas tardes", "buen dia", "buenas", "hola", "que tal"]

def get_lemma(word):
  return WordNetLemmatizer().lemmatize(word)

def prepare_for_lda(comment):
  tokens = tokenize(comment)
  tokens = [token for token in tokens if token not in es_stop]
  tokens = [token for token in tokens if len(token) > 4]
  #tokens = [token for token in tokens if token not in terms_dictionary]
  tokens = [get_lemma(token) for token in tokens]  
  #st.write(tokens) #uncomment to show generated tokens
  # return tokens
  text_data.append(tokens)

comments = last_1000_comments.apply(clean_text)
# st.table(comments.head(100))
text_data = []
comments.apply(prepare_for_lda, text_data)
# st.table(text_data[0:10])

# generate dictionary from tokens obtained from comments
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

# save dictionary and corpus
pickle.dump(corpus, open("corpus.pkl", "wb"))
dictionary.save("dictionary.gensim")

limit=50
start=2
step=6
x = range(start, limit, step)

# Execution of several models to compare the coherence value
def calculate_coherence():
  model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=text_data, start=start, limit=limit, step=step)
  # start_time = time.time()
  # st.write("--- %s seconds for compute_coherence_values ---" % (time.time() - start_time))
  return model_list, coherence_values

def show_graph(coherence_values):
  df = pd.DataFrame({'Num Topicos':x, 'Valor Coherencia':coherence_values})
  c = alt.Chart(df).mark_line().encode(x='Num Topicos', y='Valor Coherencia')
  st.altair_chart(c, use_container_width=True)

def generate_visualization(model_list):
  #FIXME I should look for the max value in model_list to get optimal_model
  optimal_model = model_list[4] #optimal value is 26, 4th pos in model_list
  topics = optimal_model.print_topics(num_topics=-1, num_words=5)
  t = []
  for topic in topics:
    t.append(topic[1].split("+"))
    #pprint(t)
  sent_topics_df = pd.DataFrame(data=t,columns=["word1","word2","word3","word4","word5"])
  st.write(sent_topics_df)

  lda_display = pyLDAvis.gensim.prepare(optimal_model, corpus, dictionary, sort_topics=False)
  pyLDAvis.display(lda_display)
  #uncomment next line if you want to make an html file with the visualization
  pyLDAvis.save_html(lda_display, 'lda.html')


model_list, coherence_values = calculate_coherence()
show_graph(coherence_values)

# Print the coherence scores
for m, cv in zip(x, coherence_values):
  st.write("Num Topics =", m, " has Coherence Value of", round(cv, 4))

generate_visualization(model_list)