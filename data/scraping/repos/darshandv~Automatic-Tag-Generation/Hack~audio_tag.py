import matplotlib.pyplot as plt
import gensim
import numpy as np
import spacy
import json
from time import time

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim
from pprint import pprint

import os, re, operator, warnings
warnings.simplefilter("ignore", DeprecationWarning)
start = time()
def clean(text):
    return str(text)

def build_texts(fname):
    """
    Function to build tokenized texts from file
    
    Parameters:
    ----------
    fname: File to be read
    
    Returns:
    -------
    yields preprocessed line
    """
    yield gensim.utils.simple_preprocess(fname, deacc=True, min_len=3)
    
def process_speech(path,file_format):
    import speech_recognition as sr
    from pydub import AudioSegment

    sound = AudioSegment.from_file(str(path), format=str(file_format))
    r = sr.Recognizer()
    article_list = []

    for i in range(0,int(len(sound)/1000),15):
        if(i+15<int(len(sound)/1000)):
#             print(i)
            cropped = sound[i*1000:(14+i)*1000]
            cropped.export("file.wav", format="wav")
            with sr.AudioFile('file.wav') as source:
                audio = r.record(source)
            try:
                text = r.recognize_google(audio)
    #             article.join(' '+str(text))
                article_list.append(text)
            except :
                text = r.recognize_sphinx(audio)
                article_list.append(text)
    #             article.join(' '+str(text))

    cropped = sound[i*1000:int(len(sound))]
    cropped.export("file.wav", format="wav")
    with sr.AudioFile('file.wav') as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
    #             article.join(' '+str(text))
        article_list.append(text)
    except :
    #     text = r.recognize_sphinx(audio)
    #     article_list.append(text)
        print('Error')
    string = ''
    for crop in article_list:
        string=string+' '+crop
    return string

def ret_top_model(threshold,corpus,dictionary,texts):
    """
    Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until this threshold is crossed. 
    
    Returns:
    -------
    lm: Final evaluated topic model
    top_topics: ranked topics in decreasing order. List of tuples
    """
    top_topics = [(0, 0)]
    while top_topics[0][1] < threshold:
        lm = LdaModel(corpus=corpus, id2word=dictionary)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=texts, dictionary=dictionary, window_size=10)
            coherence_values[n] = cm.get_coherence()
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
    return lm, top_topics

def return_keywords(data):
    nlp = spacy.load("en_core_web_lg")
    for word in nlp.Defaults.stop_words:
        lexeme = nlp.vocab[word]
        lexeme.is_stop = True
    explicit = dict({'cum dumpster':82, 'felch':82, 'cunt':82, 'skullfuck':82, 'Alabama hot pocket':82, 'cock-juggling thundercunt':82,'rusty trombone': 82,'blumpkin':82,'Cleveland S-teamer':82,'cum guzzling cock sucker':81,'glass bottom boat':81,"suck a fat baby's dick":81,'skermit':80,'fucking pussy':80,'meat flap':80,'fuck hole':80,'hairy axe wound':79,'up the ass':79, 'assmucus':79,'cumdump':79, 'beef curtain':79, 'moose nuckle':79,'cum chugger':78,'mother fucker':78, 'motherfucking':78, 'roast beef curtains':78, 'fuck':78, 'Roman Helmet':78, 'dick':78,'get some squish':77, 'eat a dick':77, 'clitty litter':77, 'eat hair pie':77, 'bisnotch':77, 'yard cunt punt':77, 'blue waffle':77, 'fist fuck':77, 'bitchass mother fucker':77,'fuck me in the ass with no Vaseline':77,'fuck yo mama':77,'chota bags':77, 'cuntee':77, 'motherfucker':77, 'meat drapes':77,'schlong juice':76, 'bang':76, 'meat tulips':76,     'cum freak':76, 'buggery':76, 'cuntsicle':76,     'fuckmeat':76, 'bust a load':76, 'butt fuck':76, 'GMILF':76, 'cock snot':76, 'shit fucker':76, 'sausage queen':76, 'fucktoy':76, 'dick hole':76, 'cock pocket':76, 'lick my froth':76, 'cunt-struck':76, 'cockbag':76,  'gangbang':75, 'pussy fart':75, 'ham flap':75, 'cum guzzler':75, 'squeeze a steamer':75, 'ass fuck':75, 'hoitch':75, 'cunt hole':75, 'clit licker':75, 'anal impaler':75, 'dick sucker':75, 'baby arm':75, 'smoke a sausage':75, 'Cuntasaurus rex':75, 'cunt face':75, 'buckle buffer':75,     'slich':75, 'fubugly':   75,     'man chowder':  75,     'key hole':  75,     'cocksucker':  75,     'get redwings':  75,     'hemped up':  75,     'smoke pole' :  75,     'like fuck' : 75,     'feedbag material':  75,     'eat fur pie':  74,     'analconda': 74 ,    'soggy muffin' : 74,     'suck a dick' : 74, 'nut butter':   74 ,    'fuck-bitch':  74 ,    "pull (one's) dick" :  74,     'get brain':  74  ,   'sweet dick daddy with the candy balls' : 74, 'get in pants':  74  ,   'felcher':  74  ,   'fuck puppet' : 74})
    doc = nlp(clean(data))
    texts, article = [], []
    explicit_content=[]
    explicit_score=0
    explicit_score_i=0
    for w in doc:
        if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and not (w.text in explicit.keys()):
            article.append(w.lemma_)
        if w.text == '\n':
            texts.append(article)
            article = []
    for bad in explicit.keys():
        if bad in str(doc):
            explicit_score+=explicit[bad]
            explicit_score_i += 1
    if explicit_score_i:
        avg_score = explicit_score/explicit_score_i 
    if not len(texts):
        articles = []
        phrases = gensim.models.phrases.Phrases(article)
        bigram = gensim.models.phrases.Phraser(phrases)
        articles.append(bigram[article])
        dictionary = Dictionary(articles)
        corpus = [dictionary.doc2bow(text) for text in articles]
        texts = articles
    else:
        phrases = gensim.models.phrases.Phrases(texts)
        bigram = gensim.models.phrases.Phraser(phrases)
        texts = [bigram[line] for line in texts]
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
    train_texts = list(build_texts(data))
    lm, top_topics = ret_top_model(0.5,corpus,dictionary,texts)
    lda_lsi_topics = [[word for word, prob in lm.show_topic(topicid)] for topicid, c_v in top_topics]
    topics=set()
    for topic in lda_lsi_topics:
        for string in topic:
            topics.add(string)
    key_tags = []
    for key in topics:
        key_tags.append(key)
    for e in doc.ents:
        if e.text.lower() not in key_tags:
            key_tags.append(e.text.lower())
    return json.dumps({'audio_key_tags':key_tags,'explicit_score':avg_score})
data = process_speech("video.mp4","mp4")
data = data + 'cunt '+'dick'
print('Speech processing time : ',time()-start)
print(return_keywords(data))
print(time()-start)