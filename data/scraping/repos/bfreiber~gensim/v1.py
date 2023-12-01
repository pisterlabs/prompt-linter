# [0] Decide on data/problem (insustry, comeptitor set, urls, start dates)
#### Industry
#### Competitor set
#### Questions (merger, new entrants, evolution of categorizaiton [spexialization], monopoly + rate of change [competition --> change?], mimicability [can you mimic competitors? what prompts you to change behavior - poor web traffic?])
# [1] Get data (urls/datetimes available from wayback machine, raw text from url)
#### Setup virtualenv + requests, beautifulsoup(?) + time?
#### Build corpus
# [2] Perform analysis (gensim, word embedding etc.)
# [3] Build materials to hand in (graphs, raw code, etc.)

################################ [1] Start Get Data ################################
# https://blog.technavio.com/blog/top-10-largest-pet-food-manufacturers
# Note to self: May want to just use dog food subsite as more relevant text?
competitor_set = [
    {'name':'Blue Buffalo', 'url':'https://bluebuffalo.com/', 'start_date':, '?'},
    {'name':'Hills Pet Food', 'url':'https://www.hillspet.com/', 'start_date':, '?'},
    ..
    {'name':'Pedigree', 'url':'https://www.pedigree.com/', 'start_date':, '?'},
    {'name':'Whiskas', 'url':'https://www.whiskas.com/', 'start_date':, '?'},#Cats only
    {'name':'Royal Canin', 'url':'https://www.royalcanin.com/us', 'start_date':, '?'},
    {'name':'Purina', 'url':'https://www.purina.com/', 'start_date':, '?'},
    {'name':'Rachael Ray Nutrish', 'url':'https://www.nutrish.com/', 'start_date':, '?'},#
    {'name':'Dads', 'url':'https://www.dadspetfoods.com/', 'start_date':, '?'},
    {'name':'Beaphar', 'url':'https://www.beaphar.com/', 'start_date':, '?'},# Beyond only food - primarily pet care, may be good to omit


    ]

# https://www.statista.com/statistics/194706/leading-us-razor-brands-in-2013-based-on-sales/#:~:text=When%20it%20comes%20to%20choosing,million%20U.S.%20dollars%20in%20sales.
competitor_set_razorblades = [
    {'name':'Harrys', 'url':'https://www.harrys.com', 'start_date':'2013'},
    {'name': 'Dollar Shave Club', 'url':'https://www.dollarshaveclub.com/', 'start_date':'2012'},
    {'name':'Gillette', 'url':'https://gillette.com/', 'start_date':'2000'},
    {'name':'Schick', 'url':'https://www.schick.com/', 'start_date':'2000'}
    ]
# Goal: Cluster OVER TIME, and see rate of change over time - hypothesis is that rate of change will increase as competitors enter, want to see this
# Women's OR international as DnD

competitor_set = [{'name':'Schick', 'url':'https://www.schick.com/', 'start_date':'2000'}]
competitor_set = [{'name':'Netflix', 'url':'https://www.netflix.com/', 'start_date':'2005'}]


def get_recent_snapshots_wayback(url, start_year=2017, end_year=2020):
    # (purina.com) --> (['http://web.archive.org/web/20170107053938/https://www.purina.com/', ..., 'http://web.archive.org/web/20170107053951/http://www.purina.com/'])
    # https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server
    import json, requests
    #url = 'purina.com'
    #url = 'http://web.archive.org/cdx/search/cdx?url='+url+'&output=json&limit=10&from=2017&to=2020'
    url = 'http://web.archive.org/cdx/search/cdx?url='+url+'&output=json&limit=none&from='+str(start_year)+'&to='+str(end_year)
    r = requests.get(url)
    json_object = r.json()
    #recent_snapshots_wayback = ['http://archive.org/wayback/available?url='+el[json_object[0].index('original')]+'&timestamp='+el[json_object[0].index('timestamp')] for el in json_object[1:]]
    recent_snapshots_wayback = ['http://web.archive.org/web/'+el[json_object[0].index('timestamp')]+'/'+el[json_object[0].index('original')] for el in json_object[1:]]
    # url = 'http://archive.org/wayback/available?url=example.com&timestamp='+str(timestamp)
    return recent_snapshots_wayback

def build_psuedo_random_wayback_subset(recent_snapshots_wayback, max_snapshots_per_month=10):
    # YYYYMMDDhhmmss
    # Take a random n=1 per month
    import random
    psuedo_random_wayback_subset = []
    years = sorted(list(set([int(snapshot[snapshot.index('web/')+len('web/'):snapshot.index('/http')][:len('YYYY')]) for snapshot in recent_snapshots_wayback])))
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    for year in years:
        for month in months:
            #relevant_snapshots = [snapshot for snapshot in recent_snapshots_wayback if (year==int(snapshot[snapshot.index('web/')+len('web/'):snapshot.index('/http')][:len('YYYY')])) and (month==((snapshot[snapshot.index('web/')+len('web/'):snapshot.index('/http')][len('YYYY'):len('YYYYMM')])))]
            relevant_snapshots = [snapshot for snapshot in recent_snapshots_wayback if (year==int(snapshot[snapshot.index('web/')+len('web/'):snapshot.index('/http')][:len('YYYY')])) and (month==((snapshot[snapshot.index('web/')+len('web/'):snapshot.index('/http')][len('YYYY'):len('YYYYMM')])))]
            #random_subset = random.sample(relevant_snapshots, min(1, len(relevant_snapshots)))
            random_subset = random.sample(relevant_snapshots, min(max_snapshots_per_month, len(relevant_snapshots)))
            for i in random_subset:
                psuedo_random_wayback_subset.append(i)
    return psuedo_random_wayback_subset

def grab_raw_text_from_url(url):
    # (https://example.com) --> ("abc def geh ...")
    import requests
    from bs4 import BeautifulSoup
    #url = 'https://www.thefarmersdog.com/'
    page = requests.get(url)
    import json
    soup = BeautifulSoup(page.text, 'html.parser')
    document = soup.html.find_all()
    #text_vector = list(set([el.text for el in document]))
    text_vector = list(set([el.text for el in document if el.name not in ['footer']]))
    text = ' '.join(list(set([el.text for el in document])))
    text = text.replace('\n','')
    return text

def grab_raw_meta_text_from_url(url):
    # (https://example.com) --> ("abc def geh ...") [title, meta title meta keywords]
    import requests
    from bs4 import BeautifulSoup
    #url = 'https://www.thefarmersdog.com/'
    page = requests.get(url)
    import json
    soup = BeautifulSoup(page.text, 'html.parser')
    title = soup.find('title').text
    if soup.find('meta', attrs={'name': 'keywords'}) != None:
        keywords = soup.find('meta', attrs={'name': 'keywords'})['content']
    else:
        keywords = ''
    description = soup.find('meta', attrs={'name': 'description'})['content']
    text = ' '.join(list(set([title.lower(), keywords.lower(), description.lower()])))
    return text


def build_corpus(competitor_set, start_year=2005, end_year=2018, max_snapshots_per_month=5):
    # () --> ()
    # We treat each time-instance of the landing page (full-text) as a document
    # A corpus is then the collection of such documents
    from langdetect import detect
    corpus = []
    #years = list(range(2010,2018))
    years = list(range(start_year,end_year))
    for el in competitor_set:
        print('Starting ' + el['name'])
        url = el['url']
        recent_snapshots_wayback = get_recent_snapshots_wayback(url, max(min(years),int(el['start_date'])), max(years))
        # Randomize subset per year
        psuedo_random_wayback_subset = build_psuedo_random_wayback_subset(recent_snapshots_wayback, max_snapshots_per_month=max_snapshots_per_month)
        for wayback_url in psuedo_random_wayback_subset:
            print('wayback_url ' + str(psuedo_random_wayback_subset.index(wayback_url)) + ' of ' + str(len(psuedo_random_wayback_subset)) + ' | ' + wayback_url)
            try:
                #text = grab_raw_text_from_url(wayback_url)
                text = grab_raw_meta_text_from_url(wayback_url)
            except:
                print('Error with '+wayback_url)
            if detect(text) == 'en':
                corpus.append({'brand':el['name'], 'url':wayback_url, 'text':text})
            else:
                print('Non english text detected')
    return corpus
################################ [1] End Get Data ################################

################################ [2] Start Gensim model ################################

def build_text_specific_stop_words(corpus, competitor_set):
    # [0] Build list of brand specific keyword counts
    from collections import defaultdict
    from gensim import corpora
    text_specific_stop_words = {el['name']:[] for el in competitor_set}
    keyword_vectors = {el['name']:[] for el in competitor_set}
    for brand in list(text_specific_stop_words.keys()):
        text_corpus = [el['text'] for el in corpus if el['brand']==brand]
        # Create a set of frequent words
        stoplist = set('for a of the and to in'.split(' '))
        # Lowercase each document, split it by white space and filter out stopwords
        texts = [[word for word in document.lower().split() if word not in stoplist]
                 for document in text_corpus]
        # Count word frequencies
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        # Only keep words that appear more than once
        processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
        dictionary = corpora.Dictionary(processed_corpus)
        keyword_vector = dictionary.token2id
        keyword_vectors[brand] = keyword_vector
    # [1] Retrun list of all keywords high in a single brand but absent from others

    return text_specific_stop_words

def build_gensim_model(corpus):
    text_corpus = [el['text'] for el in corpus]
    # Create a set of frequent words
    stoplist = set('for a of the and to in'.split(' '))
    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in text_corpus]
    # Count word frequencies
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    # Only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    #pprint.pprint(processed_corpus)

    from gensim import corpora
    dictionary = corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    from gensim import models
    # train the model
    tfidf = models.TfidfModel(bow_corpus)
    # transform the "system minors" string
    words = "dog food".lower().split()
    print(tfidf[dictionary.doc2bow(words)])
    from gensim import similarities
    index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)]

    query_document = 'dog food'.split()
    query_bow = dictionary.doc2bow(query_document)
    sims = index[tfidf[query_bow]]
    print(list(enumerate(sims)))

    return

################################ [2] End Gensim model ################################

## Need to avoid words common only to one text ##
## Build common clusters of words (e.g., affordable; x, y) [categories]
## Predict probability of each class (vector basis) [probability of existence in a given category]
## Change over time - esp pre-post introduction of competitors [change over time]

## Just one brand change over time, before and after [e.g., gillette US/CA over time vs. xyz (don't need to deal with multi-brand shennanigans)]
## [0] Build corpus (10/month, 2010-2018)
## [1] Build topics/categories
## [2] Predict probability of each category in any given time period
## [3] Graph evolution pre-post intro of new competitors (Harry's, Dollar Shave Club)

################################ [2] Start topic model ################################
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

def build_model(corpus):
    # nltk.download('stopwords')
    # python3 -m spacy download en
    import nltk
    import re
    import numpy as np
    import pandas as pd
    from pprint import pprint
    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel
    # spacy for lemmatization
    import spacy
    # Plotting tools
    import pyLDAvis
    import pyLDAvis.gensim  # don't skip this
    import matplotlib.pyplot as plt
    #%matplotlib inline
    # Enable logging for gensim - optional
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # [5] NLTK Stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['facebook', 'twitter'])
    # [8] Tokenize keywords
    from pandas import DataFrame
    df = DataFrame([el['text'] for el in corpus],columns=['Landing page'])
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    #data = df.content.values.tolist()
    data_words = list(sent_to_words(corpus))#11:20-11:22
    # [9] Create bigram/trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    # See trigram example
    print(trigram_mod[bigram_mod[data_words[0]]])
    # [10] Remove stopwords
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    ## Call functions
    #### Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    #### Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    #### Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    ##### python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    ##### Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])#11:43-
    print(data_lemmatized[:1])
    # [11] Create dictionary/corpus needed
    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus2 = [id2word.doc2bow(text) for text in texts]
    print(corpus2[:1])
    # [12] Build topic model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus2,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus2]
    # [14] How good is a model
    ## Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus2))  # a measure of how good the model is. lower the better.
    ## Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    # [16] Mallet (better LDA model)
    mallet_path = 'mallet-2.0.8/bin/mallet'
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus2, num_topics=20, id2word=id2word)
    pprint(ldamallet.show_topics(formatted=False))





# Perhaps descriptions of products etc. more meaningful?
#### PIVOTS (how to identify) + map to web traffic [can we identify pivots and/ or rate of change?]
#### Rate of change --> innovation? [titles only?]

# [all startups][successful startups]-->[rate of change] difference?
# title + meta name='DESCRIPTION' + meta name='KEYWORDS'


################################ [2] End topic model ################################

## Next steps
#### Seems like the keywords in near 100% of elements dominate - so we likely need to find a way to differentiate. Possibilities:
## [a] Custom code to remove keywords that appear in every document (tfid vectorizer?)
## [b] Rolling windows - e.g., same analysis run on first n=50 snapshots (1 year?) --> top 5 categories --> see which categories "drop out"

################################ [3] Start build startup list ################################
"""
def build_company_list():
    import requests
    import json
    from bs4 import BeautifulSoup
    url = 'https://nycstartups.net/'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    elements = soup.findAll("div", {"class": "card startup"})
    company_list = [{'name': element['data-name'], 'link':element.findChildren("a" , recursive=False)[0]['href']} for element in elements]
    return company_list

def build_company_list_ngos():
    import requests
    import json
    from bs4 import BeautifulSoup
    url = 'https://www.raptim.org/top-100-global-ngos/'
    r = requests.get(url)
    document = soup.html.find_all("article", {"class":"right"})[0]
    previous_element = ''
    for el in document:
        print(el.type)
        if type(el) == "<class 'bs4.element.Tag'>":
            print(el)
    current_element = el
        if previous_element[:4] == '<h2>':
            print('text')
        previous_element = el

    elements = document.findAll("h2")
    elements = document.findAll("p")
    soup = BeautifulSoup(r.text, 'html.parser')
    return company_list_ngos

def build_corpus(company_list):
    corpus = []
    for company in company_list:
        print(str(company_list.index(company)) + ' of ' + str(len(company_list)))
        try:
            text = grab_raw_meta_text_from_url(company['link'])
            corpus.append({'name':company['name'], 'link':company['link'], 'text':text})
        except:
            print('error with ' + company['link'])
    return corpus

r = requests.get('https://raw.githubusercontent.com/ali-ce/datasets/master/Y-Combinator/Startups.csv')

def read_csv(file_name):
    import csv
    with open(file_name) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        csv_data_rows = [row for row in reader]
    return csv_data_rows
"""
def build_company_list():
    file_name = 'startups.csv'
    csv_data_rows = read_csv(file_name)
    company_list = [csv_data_rows[0]] + [row for row in company_list[1:] if row[csv_data_rows[0].index('Website')]!='']
    return company_list

def build_corpus(company_list):
    corpus = []
    for row in company_list[1:]:
        print(str([el[0] for el in company_list[1:]].index(row[0])) + ' of ' + str(len(company_list[1:])))
        try:
            text = grab_raw_meta_text_from_url(row[company_list[0].index('Website')])
            corpus.append({'name':row[company_list[0].index('Company')], 'description': row[company_list[0].index('Description')], 'url':row[company_list[0].index('Website')], 'text':text})
        except:
            print('error with ' + row[company_list[0].index('Website')])
    return corpus
################################ [3] End build startup list ################################

################################ [4] Start topic modeling 4.0 - TFIDF ################################
def get_snapshots_wayback_at_time(url, year=2017, end_year=2020):
    # (purina.com) --> (['http://web.archive.org/web/20170107053938/https://www.purina.com/', ..., 'http://web.archive.org/web/20170107053951/http://www.purina.com/'])
    # https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server
    import json, requests
    #url = 'purina.com'
    #url = 'http://web.archive.org/cdx/search/cdx?url='+url+'&output=json&limit=10&from=2017&to=2020'
    url = 'http://web.archive.org/cdx/search/cdx?url='+url+'&output=json&limit=none&from='+str(start_year)+'&to='+str(end_year)
    r = requests.get(url)
    json_object = r.json()
    #recent_snapshots_wayback = ['http://archive.org/wayback/available?url='+el[json_object[0].index('original')]+'&timestamp='+el[json_object[0].index('timestamp')] for el in json_object[1:]]
    recent_snapshots_wayback = ['http://web.archive.org/web/'+el[json_object[0].index('timestamp')]+'/'+el[json_object[0].index('original')] for el in json_object[1:]]
    # url = 'http://archive.org/wayback/available?url=example.com&timestamp='+str(timestamp)
    return recent_snapshots_wayback

def build_psuedo_random_wayback_subset(recent_snapshots_wayback, max_snapshots_per_month=10):
    # YYYYMMDDhhmmss
    # Take a random n=1 per month
    import random
    psuedo_random_wayback_subset = []
    years = sorted(list(set([int(snapshot[snapshot.index('web/')+len('web/'):snapshot.index('/http')][:len('YYYY')]) for snapshot in recent_snapshots_wayback])))
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    for year in years:
        for month in months:
            #relevant_snapshots = [snapshot for snapshot in recent_snapshots_wayback if (year==int(snapshot[snapshot.index('web/')+len('web/'):snapshot.index('/http')][:len('YYYY')])) and (month==((snapshot[snapshot.index('web/')+len('web/'):snapshot.index('/http')][len('YYYY'):len('YYYYMM')])))]
            relevant_snapshots = [snapshot for snapshot in recent_snapshots_wayback if (year==int(snapshot[snapshot.index('web/')+len('web/'):snapshot.index('/http')][:len('YYYY')])) and (month==((snapshot[snapshot.index('web/')+len('web/'):snapshot.index('/http')][len('YYYY'):len('YYYYMM')])))]
            #random_subset = random.sample(relevant_snapshots, min(1, len(relevant_snapshots)))
            random_subset = random.sample(relevant_snapshots, min(max_snapshots_per_month, len(relevant_snapshots)))
            for i in random_subset:
                psuedo_random_wayback_subset.append(i)
    return psuedo_random_wayback_subset

################################ [4] End modeling 4.0 ################################

# Optimal
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    mallet_path = 'mallet-2.0.8/bin/mallet'
    for num_topics in range(start, limit, step):
        print('Starting ' + str(num_topics))
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus2, texts=data_lemmatized, start=10, limit=150, step=10)
import matplotlib.pyplot as plt
limit=150; start=10; step=10;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.title('2015')
plt.show()

def get_single_snapshot_wayback(url, year=2015):
    import json, requests
    url = 'http://web.archive.org/cdx/search/cdx?url='+url+'&output=json&limit=none&from='+str(year)
    r = requests.get(url)
    json_object = r.json()
    recent_snapshots_wayback = ['http://web.archive.org/web/'+el[json_object[0].index('timestamp')]+'/'+el[json_object[0].index('original')] for el in json_object[1:]]
    return recent_snapshots_wayback[0]

def build_corpus_past(corpus_present):
    corpus_past = []
    for el in corpus_present:
        print(str(corpus_present.index(el)) + ' of ' + str(len(corpus_present)))
        url = el['url']
        try:
            single_snapshot_wayback_url = get_single_snapshot_wayback(url, year=2015)
            text = grab_raw_meta_text_from_url(url)
            corpus_past.append({'name':el['name'], 'description':el['description'], 'url':single_snapshot_wayback_url, 'text':text})
        except:
            print('Error with: ' + url)
    return corpus_past
