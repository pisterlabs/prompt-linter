from tqdm import tqdm
import os
import regex as re
import pandas as pd


# topic modeling packages
import gensim
from gensim import models, corpora
from gensim.models.coherencemodel import CoherenceModel


'''
Topic Modeling Vars
'''
NUM_TOPICS = 5
UPDATE_EVERY = 10
CHUNKSIZE = 100
PASSES = 10
topic_model_settings = [{'num-topics':15,
             'parameters':{'random_state':100, 'update_every':UPDATE_EVERY, 'chunksize':CHUNKSIZE, 'passes':PASSES, 'alpha':'auto', 'per_word_topics':False}}, {'num-topics':5,
             'parameters':{'random_state':100, 'update_every':UPDATE_EVERY, 'chunksize':CHUNKSIZE, 'passes':PASSES, 'alpha':'auto', 'per_word_topics':False}}, {'num-topics':10,
             'parameters':{'random_state':100, 'update_every':UPDATE_EVERY, 'chunksize':CHUNKSIZE, 'passes':PASSES, 'alpha':'auto', 'per_word_topics':False}}]

def load_bookcorpus(idx=1):
    '''
    idx: default is 1. 0 is for original tokens. 1 is for lemma.
    '''
    bookcorpus = []
    print("Loading Data")
    DATAFILE = '../data/bookcorpus.processed.txt'
    with open(DATAFILE,'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip().split(' ')
            line = ' '.join([w.split('|')[idx] for w in line if len(w.split('|')) == 3])

            bookcorpus.append(line)
            
        
#     bookcorpus = [line.strip() for line in open(DATAFILE,'r')]
#     bookcorpus = [line.split(' ') for line in bookcorpus]
#     bookcorpus = [' '.join([w.split('|')[0] for w in l]) for l in bookcorpus]
    return bookcorpus

def load_data(lang='English'):
    with open("./Data/data-topic-modeling/English", 'r') as f:
        lines = f.readlines()
    return lines

def load_stopwords():
    # use the next line to use the NLTK stopwords
    # STOPWORDS = stopwords.words('english')
    STOPWORDSFILE = '../data/stopwords.txt'
    # load the stopwords
    STOPWORDS = []
    if(os.path.isfile(STOPWORDSFILE)):
        stopwordshandle = open(STOPWORDSFILE, 'r')
        STOPWORDS = stopwordshandle.readlines();
        STOPWORDS = [line.rstrip("\n") for line in STOPWORDS]   #remove the trailing '\n'
    else:
        print("Stopwords file not found. No stopwords are being used")
    return STOPWORDS

def clean_text(text, STOPWORDS):
#     tokenized_text = word_tokenize(text.lower())
    tokenized_text = text.split(' ')
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text


def train_bigram_model(tokenized_data, min_count=5, threshold=20):
    '''
    Train bigram model
    '''
    # Build the bigram and trigram models
    bigram_phrases = gensim.models.Phrases(tokenized_data, min_count=min_count, threshold=threshold) # higher threshold fewer phrases.
    bigram_model = gensim.models.phrases.Phraser(bigram_phrases)
    return bigram_model, bigram_phrases

def train_trigram_model(tokenized_data, bigram_phrases, min_count=5, threshold=20):
    '''
    Train bigram model
    '''
    # Build the bigram and trigram models
    trigram = gensim.models.Phrases(bigram_phrases[tokenized_data], threshold=threshold)
    trigram_model = gensim.models.phrases.Phraser(trigram)
    return trigram_model

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in tqdm(texts)]
def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in tqdm(texts)]

def preview_trigram_changes_old():
    '''
    Show some select trigrams. From bigram model where min_count = 5, threshold = 20, and trigram model where threshold = 20.
    '''
#     interesting_trigrams = [620, 689, 856, 1472, 2478, 3463, 3568, 4647, 5715, 7229, 7242]
#     trigram_changes = []
#     for i in interesting_trigrams:
#         trigram_changes.append("Original line: {}\nLine with bigrams: {}\n".format(data_words_bigrams[i], data_words_trigrams[i]))
#     return trigram_changes

    trigram_changes = ["Original line: ['council', 'quick', 'accuse', 'practicing', 'dark', 'magic']\nLine with bigrams: ['council', 'quick', 'accuse', 'practicing', 'dark_magic']\n", "Original line: ['junnie', 'asked', 'low', 'voice']\nLine with bigrams: ['junnie', 'asked', 'low_voice']\n", "Original line: ['cleared_throat', 'thinking', 'made', 'sense', 'discomfort', 'clear', 'started', 'move', 'stood', 'brought', 'closer']\nLine with bigrams: ['cleared_throat', 'thinking', 'made_sense', 'discomfort', 'clear', 'started', 'move', 'stood', 'brought', 'closer']\n", "Original line: ['make', 'matters_worse', 'run']\nLine with bigrams: ['make_matters_worse', 'run']\n", "Original line: ['backed', 'sat', 'cross_legged', 'ground']\nLine with bigrams: ['backed', 'sat_cross_legged', 'ground']\n", "Original line: ['drew_sharp', 'breath', 'words']\nLine with bigrams: ['drew_sharp_breath', 'words']\n", "Original line: ['noticed', 'chevelle', 'glaring', 'clamped_mouth', 'shut']\nLine with bigrams: ['noticed', 'chevelle', 'glaring', 'clamped_mouth_shut']\n", "Original line: ['felt', 'light', 'pressure', 'forehead', 'eyes', 'flew_open', 'instinctively', 'response', 'hours', 'struggled', 'force', 'open']\nLine with bigrams: ['felt', 'light', 'pressure', 'forehead', 'eyes_flew_open', 'instinctively', 'response', 'hours', 'struggled', 'force', 'open']\n", "Original line: ['tilted_head', 'side', 'stared', 'distance']\nLine with bigrams: ['tilted_head_side', 'stared', 'distance']\n", "Original line: ['heart_skipped', 'beat']\nLine with bigrams: ['heart_skipped_beat']\n", "Original line: ['side', 'door_swung', 'open', 'silently', 'wrapped_arm', 'waist']\nLine with bigrams: ['side', 'door_swung_open', 'silently', 'wrapped_arm_waist']\n"]
    for c in trigram_changes:
        print(c) 


def preview_trigram_changes(data_words_bigrams, data_words_trigrams):
    '''
    Show some select trigrams. From bigram model where min_count = 5, threshold = 20, and trigram model where threshold = 20.
    '''
    interesting_trigrams = []
    trigram_changes = []
    for i in interesting_trigrams:
        trigram_changes.append("Original line: {}\nLine with bigrams: {}\n".format(data_words_bigrams[i], data_words_trigrams[i]))
#     return trigram_changes

    for c in trigram_changes:
        print(c) 
        
def show_ngrams(before, after):
    df = {'Before':[], 'After':[]}
    for i, (o_text, b_text) in enumerate(list(zip(before, after))):
        if len(o_text) != len(b_text):
            for word in b_text:
                if word not in o_text and word not in df['After']:
                    df['After'].append(word)
                    df['Before'].append(word.replace('_',' '))
    df = pd.DataFrame(df, columns=['Before','After'])
    return df
      
  
        
        
def preview_bigram_changes(tokenized_data=[], data_words_bigrams=[]):
    '''
    Show some select bigrams. From bigram model where min_count = 5, threshold = 20.
    '''
    interesting_bigrams = [154, 405, 203]
    bigram_changes = []
    for i in interesting_bigrams:
        bigram_changes.append("Original line: {}\nLine with bigrams: {}\n".format(tokenized_data[i], data_words_bigrams[i]))
#     return bigram_changes

    for c in bigram_changes:
        print(c)    

def preview_bigram_changes_old(tokenized_data=[], data_words_bigrams=[]):
    '''
    Show some select bigrams. From bigram model where min_count = 5, threshold = 20.
    '''
#     interesting_bigrams = [231, 239, 253, 406, 510, 801, 811, 831]
#     bigram_changes = []
#     for i in interesting_bigrams:
#         bigram_changes.append("Original line: {}\nLine with bigrams: {}\n".format(tokenized_data[i], data_words_bigrams[i]))
#     return bigram_changes

    bigram_changes = ["Original line: ['turned', 'outstretched', 'hand', 'palm', 'indicating', 'stool', 'intention']\nLine with bigrams: ['turned', 'outstretched_hand', 'palm', 'indicating', 'stool', 'intention']\n", "Original line: ['tilted', 'head', 'nod']\nLine with bigrams: ['tilted_head', 'nod']\n", "Original line: ['gritted', 'teeth', 'block', 'irritating', 'sound']\nLine with bigrams: ['gritted_teeth', 'block', 'irritating', 'sound']\n", "Original line: ['awoke', 'bed', 'room', 'dimly', 'lit', 'single', 'flame', 'suspended', 'table']\nLine with bigrams: ['awoke', 'bed', 'room', 'dimly_lit', 'single', 'flame', 'suspended', 'table']\n", "Original line: ['single', 'flame', 'flickered', 'bedside', 'table', 'walked', 'closer', 'noticed', 'package', 'bed']\nLine with bigrams: ['single', 'flame', 'flickered', 'bedside_table', 'walked', 'closer', 'noticed', 'package', 'bed']\n", "Original line: ['night', 'bugs', 'chittered', 'high', 'pitched', 'keens', 'rising', 'loss', 'light']\nLine with bigrams: ['night', 'bugs', 'chittered', 'high_pitched', 'keens', 'rising', 'loss', 'light']\n", "Original line: ['early', 'morning', 'sun', 'streaked', 'break', 'makeshift', 'door', 'lighting', 'entirety', 'hollow']\nLine with bigrams: ['early_morning', 'sun', 'streaked', 'break', 'makeshift', 'door', 'lighting', 'entirety', 'hollow']\n", "Original line: ['pressed', 'tightly', 'oak', 'tree', 'shuffled', 'sideways', 'view']\nLine with bigrams: ['pressed', 'tightly', 'oak_tree', 'shuffled', 'sideways', 'view']\n"]
    for c in bigram_changes:
        print(c)    

def print_all_bigram_differences(tokenized_data, data_words_bigrams):
    for i, (o_text, b_text) in enumerate(list(zip(tokenized_data, data_words_bigrams))):
        if len(o_text) != len(b_text):
            print("idx:", i, "Original line:", o_text, "Line with bigrams:", b_text)
                 
def show_topic_terms(lda_model, NUM_TOPICS, num_terms=10):
    
    columns = ['Topic']
    for w, pr in zip(["W{}".format(i+1) for i in range(num_terms)], ["W{} Pr".format(i+1) for i in range(num_terms)]):
        columns.append(w)
        columns.append(pr)
        
    df = {col:[] for col in columns}
    for idx in range(NUM_TOPICS):
        df['Topic'].append(idx)
        # Print the first 10 most representative topics
        terms = lda_model.print_topic(idx, num_terms)
        terms = terms.split(' + ')
        for i, t in enumerate(terms):
            parts = t.split('*')
            word = parts[1].replace('"', '')
            pr = float(parts[0])
            
            df["W{}".format(i+1)].append(word)
            df["W{} Pr".format(i+1)].append(pr)
            
    df = pd.DataFrame(df, columns=columns).set_index('Topic')
    
    return df


def name_topics(KEPT_TOPICS, KEPT_TOPIC_NAMES = {}):
    
    print("Please give a name for each Kept Topic")
    try:
        for kept_topic in KEPT_TOPICS:
            if kept_topic not in KEPT_TOPIC_NAMES:
                print("Naming Topic", kept_topic)
                print("-"*80, '\n\n')
                print("Topic terms:", '\n')
            #     print(topic_terms.loc[topic_terms['Topic'] == kept_topic])
                print(topic_terms.iloc[kept_topic, [i for i in range(20) if i%2 == 0]])

                print('\n\n')
                print("Input name for topic {}:".format(kept_topic))
                topic_name=input()
                topic_name=topic_name.strip()

                print("User input:", topic_name)
                print("\nKeep {} as the name for Topic {}? Confirm y/n".format(topic_name, kept_topic))

                confirm=input()
                confirm = confirm.strip()
                while 'n' in confirm.lower():
                    print("Input name for topic {}:".format(kept_topic))
                    topic_name=input()
                    topic_name=topic_name.strip()

                    print("User input:", topic_name)
                    print("\nKeep {} as the name for Topic {}? Confirm y/n".format(topic_name, kept_topic))
                KEPT_TOPIC_NAMES[kept_topic] = topic_name


                print('\n\nNamed Topic {} as {}'.format(kept_topic, topic_name))
                print("="*80, '\n\n\n')
    except:
        print("Something went wrong...returning current topic names.")
        return KEPT_TOPICS


    print("\n\nDone naming topics.")
    return KEPT_TOPICS

soln = models.LdaModel.load('./utils/tm/lda_model')
import pickle
import pyLDAvis.gensim
import pyLDAvis
with open("./utils/tm/corpus", 'rb') as f:
    corpus_ = pickle.load(f)
with open("./utils/tm/dictionary", 'rb') as f:
    dictionary_ = pickle.load(f)
def show_model(model=soln, corpus=corpus_, dictionary=dictionary_):
    print("Generating visual, this will take a few moments...")
    pyLDAvis.enable_notebook()
    LDAvis_prepared = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    return LDAvis_prepared
    
    
    
    
    
    
    
    
    
    
    

