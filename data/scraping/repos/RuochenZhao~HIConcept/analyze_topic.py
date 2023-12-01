from visualize import read_topics
from gensim.models.coherencemodel import CoherenceModel
import os
import gensim
from gensim.corpora import Dictionary
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer
import pandas as pd

def calc_topic_coherence(topic_words,docs,dictionary,emb_path=None,taskname=None,sents4emb=None,calc4each=False):
    # emb_path: path of the pretrained word2vec weights, in text format.
    # sents4emb: list/generator of tokenized sentences.
    # Computing the C_V score
    cv_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_v')
    cv_per_topic = cv_coherence_model.get_coherence_per_topic() if calc4each else None
    cv_score = cv_coherence_model.get_coherence()
    
    # Computing the C_W2V score
    try:
        w2v_model_path = os.path.join(os.getcwd(),'data',f'{taskname}','w2v_weight_kv.txt')
        # Priority order: 1) user's embed file; 2) standard path embed file; 3) train from scratch then store.
        if emb_path!=None and os.path.exists(emb_path):
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(emb_path,binary=False)
        elif os.path.exists(w2v_model_path):
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path,binary=False)
        elif sents4emb!=None:
            print('Training a word2vec model 20 epochs to evaluate topic coherence, this may take a few minutes ...')
            w2v_model = gensim.models.Word2Vec(sents4emb,vector_size=300,min_count=1,workers=6,epochs=2)
            keyed_vectors = w2v_model.wv
            print('Finished!')
        else:
            w2v_model = gensim.downloader.load('word2vec-google-news-300')
            keyed_vectors = w2v_model.wv
            
        w2v_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_w2v',keyed_vectors=keyed_vectors)

        w2v_per_topic = w2v_coherence_model.get_coherence_per_topic() if calc4each else None
        w2v_score = w2v_coherence_model.get_coherence()
    except Exception as e:
        print(e)
        #In case of OOV Error
        w2v_per_topic = [None for _ in range(len(topic_words))]
        w2v_score = None
    
    # Computing the C_UCI score
    c_uci_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_uci')
    c_uci_per_topic = c_uci_coherence_model.get_coherence_per_topic() if calc4each else None
    c_uci_score = c_uci_coherence_model.get_coherence()
    
    
    # Computing the C_NPMI score
    c_npmi_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_npmi')
    c_npmi_per_topic = c_npmi_coherence_model.get_coherence_per_topic() if calc4each else None
    c_npmi_score = c_npmi_coherence_model.get_coherence()
    return (cv_score,w2v_score,c_uci_score, c_npmi_score),(cv_per_topic,w2v_per_topic,c_uci_per_topic,c_npmi_per_topic)

def mimno_topic_coherence(topic_words,docs):
    tword_set = set([w for wlst in topic_words for w in wlst])
    word2docs = {w:set([]) for w in tword_set}
    for docid,doc in enumerate(docs):
        doc = set(doc)
        for word in tword_set:
            if word in doc:
                word2docs[word].add(docid)
    def co_occur(w1,w2):
        return len(word2docs[w1].intersection(word2docs[w2]))+1
    scores = []
    for wlst in topic_words:
        s = 0
        for i in range(1,len(wlst)):
            for j in range(0,i):
                s += np.log((co_occur(wlst[i],wlst[j])+1.0)/len(word2docs[wlst[j]]))
        scores.append(s)
    return np.mean(s)

# now build docs and dictionary
data = load_dataset('ag_news', cache_dir="../../hf_datasets/")
docs = data['train']['text']
tokenizer = BertTokenizer.from_pretrained(f'bert-base-uncased', do_lower_case=True)
docs = [tokenizer.tokenize(s) for s in docs]
docs = [line for line in docs if line!=[]]
# build dictionary
dictionary = Dictionary(docs)
#self.dictionary.filter_n_most_frequent(remove_n=20)
dictionary.filter_extremes(no_below=5, no_above=0.1, keep_n=None)  # use Dictionary to remove un-relevant tokens
dictionary.compactify()
dictionary.id2token = {v:k for k,v in dictionary.token2id.items()} # because id2token is empty by default, it is a bug.

concepts_to_analyze = [3, 5, 50, 100]
dict = {'layer': concepts_to_analyze, 'cv_score': [], 'w2v_score': [], 'c_uci_score': [], 'c_npmi_score': []}
# for l in layers_to_analyze:
for l in concepts_to_analyze:
    save_file = f'models/news/bert/two_stage/topics_two_stage_-1_{l}_concept.txt'
    words = read_topics(save_file)
    (cv_score,w2v_score,c_uci_score, c_npmi_score),(_,_,_,_) = calc_topic_coherence(words,docs,dictionary, sents4emb=docs)
    print(f'FOR LAYER {l}, cv_score: {cv_score}; w2v_score: {w2v_score}; c_uci_score: {c_uci_score}, c_npmi_score: {c_npmi_score}')
    dict['cv_score'].append(cv_score)
    dict['w2v_score'].append(w2v_score)
    dict['c_uci_score'].append(c_uci_score)
    dict['c_npmi_score'].append(c_npmi_score)

df = pd.DataFrame.from_dict(dict)
df.to_csv('concept_analysis.csv')
