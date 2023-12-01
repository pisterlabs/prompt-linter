#%%
#from sklearn.datasets import fetch_20newsgroups
import os, sys,ssl,argparse
sys.path.insert(0,'../../libs')
sys.path.insert(0,'../..')
sys.path.insert(0,'..')
import config
from tqdm import tqdm
import numpy as np
## in case you are behind a proxy 
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
import joblib
from joblib import Parallel, delayed
import copy, time
from utils import chunks,rename_if_exist,get_all_files,txt2list
import warnings
warnings.filterwarnings("ignore")

## topic model packages
from bertopic import BERTopic
import gensim
import pandas as pd
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from topic_evaluator import eval_coherence_score, eval_diversity_score

## import arguments 
from topic_arguments import topic_model_args
from topic_hyper_params import load_hyper_params,get_params_diff


#%%
def read_txt(f_p,min_len=20):
    txt_l = txt2list(f_p)
    txt_l = [t for t in txt_l if len(t.split())>min_len]
    return txt_l

def data_clean_for_topic(input_text:str,min_len:int):
    """
    simple data cleaning and filtering for topic model training
    """
    if isinstance(input_text):
        pass
    else:
        return ""

    input_text= input_text.replace('<Title>','')
    if len(input_text.split())>min_len:
        return input_text
    else:
        return ""
    
def read_all_data(raw_txt_folder,min_len=20,data_clean_func=None):
    out_list = []
    input_files = get_all_files(raw_txt_folder,'.txt')
    for inf in tqdm(input_files):
        txt_l = read_txt(inf,min_len)
        if len(txt_l)>1:
            if data_clean_func:
                txt_l = data_clean_for_topic(txt_l,min_len)
                if len(txt_l)>1:
                    out_list.extend(txt_l)
            else:
                out_list.extend(txt_l)

    return out_list

def model_setup(train_args):
    '''
    initialize topic model with training args 
    '''
    ## Step 3 - set up umap for reduction 
    ## if you want to make it reproducable; set random state in umap to be fixed 
    umap_model = UMAP(n_neighbors=train_args.n_neighbors,   # local neighborhood size for UMAP. default is 15, larget mean more global structure
                                                            # This is the parameter that controls the local versus global structure in data
                    n_components=train_args.n_components,   # output dimension for UMAP
                    min_dist=0,             # to allow UMAP to place points closer together (the default value is 1.0)
                    metric='cosine',        # use cosine distance 
                    random_state=42)        # fix random seed 

    ## Step 3 - Cluster reduced embeddings
    ## see link for more param selection:
    ## https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#parameter-selection-for-hdbscan
    hdbscan_model = HDBSCAN(min_cluster_size=train_args.min_cluster_size,  #the minimum number of documents in each cluster, for larger data, this should be larger
                            min_samples=train_args.min_samples,            #controls the number of outliers. It defaults to the same value as min_cluster_size. 
                                                            #The larger the value of min_samples you provide, the more conservative the clustering – more points will be declared as noise, 
                                                            #and clusters will be restricted to progressively more dense areas
                                                            #we should keep this constant when tuning other parameters 
                            metric=train_args.metric,       #I guess we can try cosine here ? 
                            cluster_selection_method='eom', #The default method is 'eom' for Excess of Mass, the algorithm described in How HDBSCAN Works https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html.
                            prediction_data=True)

    ## Step 4-5 - prepare param for c-tfidf for topic representation 
    ## additional topic merging will be done by compare distance (based on bog method on c-tfidf), set to auto will use HDBSCAN
    ## remove stop words when fingding keys for topic representation ; sbert will still use full sentence 
    vectorizer_model = CountVectorizer(ngram_range=(1, 2),
                                        stop_words="english",       # you can also provide a customized list 
                                        min_df=train_args.min_df,                  # set min number of word frequency
                                        #vacabulary=custom_vocab,   # you can also use a customized vocabulary list, 
                                                                    # e.g use keybert: https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#keybert-bertopic
                                        )               
    ## ctfidf param can be pass in topicbert main function 

    if train_args.TUNE:
        emb_model = None
    else:
        print('use {} as embeding model'.format(train_args.model_checkpoint))
        emb_model = SentenceTransformer(train_args.model_checkpoint)  

    ## call main function 
    topic_model = BERTopic(
                    umap_model=umap_model,              # Reduce dimensionality 
                    hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
                    vectorizer_model=vectorizer_model,  # Step 4,5 - use bang of words and ctfidf for topic representation
                    embedding_model=emb_model,
                    #diversity= train_args.diversity,            in 0.14, removed # Step 6 - Diversify topic words ; maybe also try 0.5?
                    ## other params 
                    language="English",
                    verbose=train_args.verbose,
                    top_n_words=train_args.top_n_words,         # number of topic words to return; can be changed after model is trained 
                                                                # https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.update_topics
                    min_topic_size=train_args.min_cluster_size, # this should be the same as min_cluster_size in HDBSCAN
                    nr_topics=train_args.nr_topics,               # number of topics you want to reduce to ; auto will use results from HDBSCAN on c-tfidf
                    calculate_probabilities = train_args.calculate_probabilities, # Whether to calculate the probabilities of all topics per document instead of the probability of the assigned topic per document. 
                    )
    
    return topic_model 

def train_topic_model(args,docs,embeddings):
    topic_model = model_setup(args)
    topics, probabilities = topic_model.fit_transform(docs,embeddings)
    if isinstance(probabilities,np.ndarray):
        probabilities = probabilities.tolist()

    return topics,probabilities,topic_model

def eval_topic_model(docs,topics,probabilities,topic_model,n_workers=1):
    coherence_scores = eval_coherence_score(docs,topics,probabilities,topic_model,n_workers=n_workers)
    topic_freq = topic_model.get_topic_freq()
    outlier_percent = topic_freq['Count'][topic_freq['Topic'] == -1].iloc[0]/topic_freq['Count'].sum()
    n_topics = len(topic_model.get_topic_freq())
    diversity_score = eval_diversity_score(topic_model)
    
    return coherence_scores,outlier_percent,n_topics,diversity_score

def train_and_eval(args,docs,embeddings,n_workers=1):
    try:
        topics,probabilities,topic_model = train_topic_model(args,docs,embeddings)
    except Exception as e:
        print('--Topic Model training error -- : {}'.format(e))
        topics,probabilities,topic_model = (None,None,None)
    
    if topic_model is not None:
        coherence_scores,outlier_percent,n_topics,diversity_score = eval_topic_model(docs,topics,probabilities,
                                                                                topic_model,n_workers=n_workers)
    else:
        coherence_scores,outlier_percent,n_topics,diversity_score = (None,None,None,None)
    
    ## you probably aldo don't want too many outliers 
    ## other than tune cluster size, you can also try reducer outliers after model is trained 
    #https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.reduce_outliers

    return coherence_scores,outlier_percent,n_topics,diversity_score

def pack_update_param(param,coherence_scores,outlier_percent,n_topics,diversity_score):
    eval_dict = {
        'coherence': coherence_scores,
        'diversity': diversity_score,
        'outlier': outlier_percent,
        'number_topics': n_topics,
    }
    if param:
        param.update(eval_dict)
        return param
    else:
        return eval_dict

def get_param_results(param,args,docs,embeddings):
    if param:
        args.__dict__.update(param)
    try:
        coherence_scores,outlier_percent,n_topics,diversity_score = train_and_eval(args,docs,embeddings)
    except Exception as e:
        print('-- Error -- \n{}\n{}'.format(param,e))
        coherence_scores,outlier_percent,n_topics,diversity_score = (None,None,None,None)
    res = pack_update_param(param,coherence_scores,outlier_percent,n_topics,diversity_score)
    return res

#%%
if __name__ == "__main__":
    #startTime = time.time()

    # args = topic_model_args(['--model_checkpoint',
    #                          '/data/chuang/Language_Model_Training_Data/Models/Saved_SBERT/10000',
    #                          '--out_folder','/data/chuang/Language_Model_Training_Data/Models/Topic_Models/step_10000',
    #                          '--result_path','/data/chuang/Language_Model_Training_Data/Models/Topic_Models/baeline/hp_tune_results.csv'
    #                          ])
    args = topic_model_args()


    #%%
    ## set paths 
    model_name = args.model_checkpoint
    data_folder = args.data_folder
    input_folder = args.input_files_folder
    out_folder = args.out_folder
    emb_path = os.path.join(out_folder,'sentence_embeddings.npy')
    docs_path = os.path.join(out_folder,'docs.npy')
    train_args_space = get_params_diff(args.hyper_param_space_path,args.result_path)
    if args.test_run:  ## test trying loop with small params 
        train_args_space = train_args_space[:200]
    result_path = rename_if_exist(args.result_path)
    if args.verbose:
        print("Results will be saved at {}".format(result_path))

    ## set up topics models 
    if not args.LOAD_EMB:
        ## read raw documents 
        docs = read_all_data(input_folder)
        ## load model 
        sentence_model = SentenceTransformer(model_name)                
        ## encode sentences 
        embeddings = sentence_model.encode(docs, show_progress_bar=True)
        assert len(docs)==len(embeddings)
        ## cache embedings 
        embeddings = np.array(embeddings)
        docs = np.array(docs)
        np.save(emb_path,embeddings)
        np.save(docs_path,docs)
    else:
        print('Load embeding from {}'.format(emb_path))
        embeddings = np.load(emb_path)
        docs = np.load(docs_path)
        assert len(docs)==len(embeddings)
        print('Number of docs: {}'.format(len(docs)))

    #%%
    ## for testing purpose, try a small sample size 
    embeddings = embeddings[:12000]
    docs = docs[:12000]
    #%%
    results = []
    if args.TUNE:
        if args.n_worker>1:
            args_copy = copy.deepcopy(args)
            number_of_cpu = args.n_worker #joblib.cpu_count() - 2 
            startTime = time.time()
            with Parallel(n_jobs=number_of_cpu,verbose=5, backend='loky') as parallel_pool:
                batched_train_args_space = list(chunks(train_args_space,args.chunk_size))
                for args_space in tqdm(batched_train_args_space):
                    delayed_funcs = [delayed(get_param_results)(p,args_copy,docs,embeddings) for p in args_space]
                    multi_res = parallel_pool(delayed_funcs)
                    results.extend(multi_res)
                    res_df = pd.DataFrame(results)
                    res_df.to_csv(result_path)  ## export every chunk 
            executionTime = (time.time() - startTime)
            print('Execution time in seconds: ' + str(executionTime))
        else:
            for idx,params in enumerate(tqdm(train_args_space)):
                args.__dict__.update(params)
                try:
                    updated_params = get_param_results(params,args,docs,embeddings)
                    results.append(updated_params)
                    if args.verbose:
                        print(updated_params)
                except Exception as e:
                    print('-- Error -- \n{}\n{}'.format(params,e))
                    results.append(params)
                ## write out every 5 steps 
                if idx%5 == 0:
                    res_df = pd.DataFrame(results)
                    res_df.to_csv(result_path)

        res_df = pd.DataFrame(results)
        res_df.to_csv(result_path)
    else:
        print(args)
        #for i in tqdm(range(1)):
        topics,probabilities,topic_model = train_topic_model(args,docs,embeddings)
        ## save model and model outputs 
        topic_model_out_path = os.path.join(args.out_folder,'topic_model')
        topic_model.save(topic_model_out_path, save_embedding_model=True)
        np.save(os.path.join(args.out_folder,'topics.npy'),np.array(topics))
        np.save(os.path.join(args.out_folder,'probabilities.npy'),np.array(probabilities))

        coherence_scores,outlier_percent,n_topics,diversity_score = eval_topic_model(docs,topics,probabilities,topic_model,n_workers=1)
        res_dict = pack_update_param(None,coherence_scores,outlier_percent,n_topics,diversity_score)
        print(res_dict)
        print('Model saved in {}'.format(args.model_checkpoint))

    # ## track executionTime
    # executionTime = (time.time() - startTime)
    # print('Execution time in seconds: ' + str(executionTime))


 ## to be improved 
 # - count vectorvizer, remove integer numbers see topic_model.vectorizer_model.get_feature_names(); many doesn't make sense
 # - more topic evaluation on coherence and diversity https://github.com/MaartenGr/BERTopic/issues/594
 #  --- https://github.com/MIND-Lab/OCTIS

# %%
