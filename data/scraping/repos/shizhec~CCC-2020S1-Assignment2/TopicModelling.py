# @Peng Cao 798530
# @Email:  caop1@student.unimelb.edu.au

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import re, numpy as np, pandas as pd
from pprint import pprint
import argparse

import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'
                  , 'today', 'great','new','https','com','love','thing','day'])
#python -m spacy download en

warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

#add parsers and use argument to have various of scripts.
parser = argparse.ArgumentParser(description = 'topic analysis')
parser.add_argument('--inputfilename', type=str, default='defaultinput.json')
parser.add_argument('--outputfilename', type=str, default='defaultoutput.json')
parser.add_argument('--modelsavename', type=str, default='defaultmodelname.json')
args = parser.parse_args()

#load data of input file
f = open(args.inputfilename)
twitter = json.load(f)

text = []
id = []
for i in range(len(twitter['docs'])):
    text.append(twitter['docs'][i]['text'])
    id.append(twitter['docs'][i]['id'])
df = pd.DataFrame({'text':text, 'id': id})

#pip install spacy

#break sentences to words, and at the same time remove some components using regular expressions.
def sent_to_words(sentences):
    for sent in sentences:
        #remove emails
        sent = re.sub('\S*@\S*\s?', '', sent)
        #remove newline chars
        sent = re.sub('\s+', ' ', sent)
        #remove single quotes
        sent = re.sub("\'", "", sent)
        #use the default gensim simple preprocess.
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  

data_words = list(sent_to_words(df['text']))

# Using the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

#define a function of process words, only takes word that are NOUN and ADJ because others do not have meanings for topic.
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ']):
    #doing the pre-process defined before.
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    #load and lemmatization
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    #remove stopwords again
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

data_ready = process_words(data_words)



#LDA needed: Create Dictionary
id2word = corpora.Dictionary(data_ready)

#LDA needed: Create Corpus using Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]

# Build LDA model, with 4 topics
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=4, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=100,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

pprint(lda_model.print_topics())

#start doing wordcloud plot.
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=800,
                  height=560,
                  max_words=30,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

#have topics of 4, plot on a 2*2 shape.
fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=16))
    plt.gca().axis('off')

#save the topic image
plt.savefig(args.outputfilename)
plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

#save the lda model
lda_model.save(args.modelsavename)







