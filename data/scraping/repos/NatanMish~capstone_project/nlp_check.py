import pandas as pd
from itertools import product
from tqdm import tqdm
import multiprocessing as mp

class nlp:
    #need to download from nltk - 'averaged_perceptron_tagger','wordnet','stopwords'
    re = __import__('re')
    pd = __import__('pandas')
    nltk = __import__('nltk')
    
    def __init__(self):
        print('nlp V.0.1 \nImported pandas,re packages')
        
    def clean_text_for_topic_modelling(self,text_column):
        text_column=text_column.str.lower()
        text_column = [self.re.sub(r'https?:\/\/*[^ ]*', '', x) for x in text_column]
        text_column = [self.re.sub(r'[.,;/]',' ', x) for x in text_column]
        text_column = [self.re.sub(r'\((cont)\)','', x) for x in text_column]
        text_column = [self.re.sub(r'[^A-Za-z0-9$% ]','', x) for x in text_column]
        text_column = [x.split() for x in text_column]
        temp_corpus=[]
        for tweet in text_column:
            new_list=[w for w in tweet if len(w)>2]
            temp_corpus.append(new_list)
        text_column=self.pd.Series(temp_corpus)
        return text_column
    
    def remove_stopwords_from_corpus(self,text_column,extra_stopwords_list=[]):
        from nltk.corpus import stopwords
        stops = set(stopwords.words('english')).union(extra_stopwords_list)
        text_column=text_column.apply(lambda x: [w for w in x if w not in stops])
        return text_column
    
    def stemm_corpus(self,text_column,stemmer='porter'):
        if stemmer=='porter':
            from nltk.stem import PorterStemmer
            porter = PorterStemmer()
            text_column=text_column.apply(lambda x:[porter.stem(w) for w in x])
        elif stemmer=='lancaster':
            from nltk.stem import LancasterStemmer
            lancaster=LancasterStemmer()
            text_column=text_column.apply(lambda x:[lancaster.stem(w) for w in x])
        else:
            from nltk.stem import SnowballStemmer
            snowball=SnowballStemmer("english")
            text_column=text_column.apply(lambda x:[snowball.stem(w) for w in x])
        return text_column
    
    def lemmatize_corpus(self,text_column):
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet
        
        def get_word_pos(word):
            tag = self.nltk.pos_tag(word)[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        
        wordnet_lemmatizer = WordNetLemmatizer()
        text_column=text_column.apply(lambda x:[wordnet_lemmatizer.lemmatize(w, pos=get_word_pos(w)) for w in x])
        return text_column
    
    def create_ngrams(self,text_column,replace=True,num_grams=2):
        from nltk import ngrams
        if replace:
            grams_list=[]
            for tweet in text_column:
                grams_list.append([' '.join(ngram) for ngram in ngrams(tweet,num_grams)])
            return self.pd.Series(grams_list)
        else:
            for i,tweet in text_column.iteritems():
                copy_tweet=tweet.copy()
                for ngram in ngrams(tweet,2):
                    copy_tweet.append(' '.join(ngram))
                text_column.at[i]=copy_tweet
            return text_column
        
    def gensim_dic(self,text_column,filter_extremes=True,no_below=3,no_above=0.95):
        from gensim.corpora import Dictionary
        dictionary=Dictionary(corpus)
        if filter_extremes:
            dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        return dictionary
    
    def bow_corpus(self,dictionary,text_column):
        return [dictionary.doc2bow(doc) for doc in text_column]
    
    def vec_to_tfidf(self,bow_corpus):
        from gensim import models
        tfidf = models.TfidfModel(bow_corpus)
        return tfidf[bow_corpus]
    
    def lda_model(self,tfidf_corp,dic,num_topics=25,passes=15, alpha=0.01,eta='auto'):
        from gensim import models
        return models.ldamodel.LdaModel(corpus=tfidf_corp, num_topics=num_topics, id2word=dic, passes=passes, alpha=alpha, 
                                             eta=eta,random_state=13)
    
    def get_coherence(self,lda_model,corpus,dic):
        from gensim.models import CoherenceModel
        coherence_model_lda = CoherenceModel(model=lda_model, texts=corpus, dictionary=dic, coherence='c_v')
        return coherence_model_lda.get_coherence()
    
    def is_tweet_intopic(self,text_column,words_list):
        
        def is_tweets_words_inlist(tweet,words_list):
            for word in tweet:
                if word in words_list:
                    return True
            return False
        
        results=[]
        for tweet in text_column:
            results.append(is_tweets_words_inlist(tweet,words_list))
        return self.pd.Series(results)
    
    def assign_lda_topic(self,lda_model,bow_corpus):
        topics=[]
        data=lda_model.get_document_topics(bow_corpus)
        for tweet in data:  
            df = self.pd.DataFrame(tweet, columns=['topic_num', 'probability'])
            topics.append(df.loc[df['probability'].idxmax(), 'topic_num'])
        return (self.pd.Series(topics))
		
		
nlpob=nlp()

df=pd.read_csv('trump_tweets_reducted.csv',parse_dates=['created_at_utc'])
df.reset_index(inplace=True,drop=True)
corpus=nlpob.clean_text_for_topic_modelling(df['text'])
corpus=nlpob.remove_stopwords_from_corpus(corpus,['realdonaldtrump','amp','president','android','iphone'])

def expand_grid(dictionary):
    return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())

dictionary = {'stemmers':['snowball','porter','lancaster'],
'lemmatize':[True,False],
'replace_ngrams':[True,False],
'num_grams':[2,3],
'dic_extremes_nobelow':[3,5,10],
'dic_extremes_noabove':[0.9,0.95,0.99],
'num_topics':[20,30],
'alpha':[0.001,0.0001,0.01]}

combos_grid=expand_grid(dictionary)

def run_by_combo(stemmers,lemmatize,replace_ngrams,num_grams,dic_extremes_nobelow,dic_extremes_noabove,num_topics,alpha):
    corpus_temp=nlpob.stemm_corpus(corpus,stemmer=stemmers)
    if lemmatize:
        corpus_temp=nlpob.lemmatize_corpus(corpus_temp)
	
    corpus_temp=nlpob.create_ngrams(corpus_temp,replace=replace_ngrams,num_grams=num_grams)
    dic=nlpob.gensim_dic(corpus_temp,no_below=dic_extremes_nobelow,no_above=dic_extremes_noabove)
    bow_corp=nlpob.bow_corpus(dic,corpus_temp)
    tfidf_corp=nlpob.vec_to_tfidf(bow_corp)
    lda_model = nlpob.lda_model(tfidf_corp,dic,num_topics=num_topics,alpha=alpha)
    return nlpob.get_coherence(lda_model,corpus_temp,dic)
    

combos=[]
for i,combo in combos_grid.iterrows():
    combos.append(combo)

pool = mp.Pool(mp.cpu_count())

results = [pool.apply(run_by_combo, args=(combo)) for combo in tqdm(combos)]

pool.close()   

pd.Series(results).to_csv('results.csv')