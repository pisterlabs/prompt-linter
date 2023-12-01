import tomotopy as tp
from gensim.models import CoherenceModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
import preprocess
def get_hdp_topics(hdp, top_n=10):
    '''Wrapper function to extract topics from trained tomotopy HDP model 
    
    ** Inputs **
    hdp:obj -> HDPModel trained model
    top_n: int -> top n words in topic based on frequencies
    
    ** Returns **
    topics: dict -> per topic, an arrays with top words and associated frequencies 
    '''
    
    # Get most important topics by # of times they were assigned (i.e. counts)
    sorted_topics = [k for k, v in sorted(enumerate(hdp.get_count_by_topics()), key=lambda x:x[1], reverse=True)]

    topics=dict()
    
    # For topics found, extract only those that are still assigned
    for k in sorted_topics:
        if not hdp.is_live_topic(k): continue # remove un-assigned topics at the end (i.e. not alive)
        topic_wp =[]
        for word, prob in hdp.get_topic_words(k, top_n=top_n):
            topic_wp.append((word, prob))

        topics[k] = topic_wp # store topic word/frequency array
        
    return topics

def train_HDPmodel(hdp, word_list, mcmc_iter, burn_in=100, quiet=False):
    '''Wrapper function to train tomotopy HDP Model object
    
    *** Inputs**
    hdp: obj -> initialized HDPModel model
    word_list: list -> lemmatized word list of lists
    mcmc_iter : int -> number of iterations to train the model
    burn_in: int -> MC burn in iterations
    quiet: bool -> flag whether to print iteration LL and Topics, if True nothing prints out
    
    ** Returns**
    hdp: trained HDP Model 
    '''
    
    # Add docs to train
    for vec in word_list:
        hdp.add_doc(vec)

    # Initiate MCMC burn-in 
    hdp.burn_in = 100
    hdp.train(0)
    print('Num docs:', len(hdp.docs), ', Vocab size:', hdp.num_vocabs, ', Num words:', hdp.num_words)
    print('Removed top words:', hdp.removed_top_words)
    print('Training...', file=sys.stderr, flush=True)

    # Train model
    step=round(mcmc_iter*0.10)
    for i in range(0, mcmc_iter, step):
        hdp.train(step, workers=3)
        if not quiet:
            print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}'.format(i, hdp.ll_per_word, hdp.live_k))
        
    print("Done\n")  
    
    return hdp
def eval_coherence(topics_dict, word_list, coherence_type='c_v'):
    '''Wrapper function that uses gensim Coherence Model to compute topic coherence scores
    
    ** Inputs **
    topic_dict: dict -> topic dictionary from train_HDPmodel function
    word_list: list -> lemmatized word list of lists
    coherence_typ: str -> type of coherence value to comput (see gensim for opts)
    
    ** Returns **
    score: float -> coherence value
    '''
    
    # Build gensim objects
    dictionary = preprocess.get_dictionary(word_list)
    corpus = preprocess.get_bow(dictionary, word_list)
    
    # Build topic list from dictionary
    topic_list= build_topic_list(topics_dict)
            

    # Build Coherence model
    print("Evaluating topic coherence...")
    
    score = get_coherence(topic_list, corpus, dictionary, word_list)
    print ("Done\n")
    return score
def get_model_topics(word_list):
    tw_list = [tp.TermWeight.ONE, # all terms weighted equally
           tp.TermWeight.PMI, # Pointwise Mutual Information term weighting
           tp.TermWeight.IDF] # down-weights high frequency terms, upweights low freq ones

    tw_names = ['one', 'pmi', 'idf']
    model_topics =[]

    for i, term_weight in enumerate(tw_list):
        hdp = tp.HDPModel(tw=term_weight, min_cf=5, rm_top=7, gamma=0.00000000001, alpha=0.00000001,
                     initial_k=10, seed=99999)
    
        print("Model " + tw_names[i] )
        hdp = train_HDPmodel(hdp, word_list, mcmc_iter=10)
        hdp.save(''.join(['hdp_model_',tw_names[i],".bin"]))
    
        model_topics.append(get_hdp_topics(hdp, top_n=10))
    return model_topics
def build_topic_list(hdp_topics):
    topic_list=[]
    for k, tups in hdp_topics.items():
        topic_tokens=[]
        for w, p in tups:
            topic_tokens.append(w)
        topic_list.append(topic_tokens)
    return topic_list

def get_coherence(topic_list, bow, dic, word_list):
    cm = CoherenceModel(topics=topic_list, corpus=bow, dictionary=dic, texts=word_list, 
                    coherence='c_v')
    print(cm.get_coherence())
    return cm.get_coherence()

def load_hdp_models():
    hdp_one = tp.HDPModel.load("hdp_model_one.bin")
    hdp_pmi = tp.HDPModel.load("hdp_model_pmi.bin")
    hdp_idf = tp.HDPModel.load("hdp_model_idf.bin")
    return hdp_one, hdp_pmi, hdp_idf

def hdp_topics_to_wordclouds(model, topic_dict, save=False):
    '''Wrapper function that generates wordclouds for ALL topics of a tomotopy model
    
    ** Inputs **
    model: obj -> tomotopy trained model
    topic_dic: dict -> per topic, an arrays with top words and associated frequencies
    save: bool -> If the user would like to save the images
    
    ** Returns **
    wordclouds as plots
    '''
    
    wcloud = WordCloud(background_color="white")
    fig, ax = plt.subplots(1, 3, figsize=(15,4))

    cnt=0
    for k, arr in topic_dict.items():
        
        create_wordcloud(model, k, fig, ax[cnt], save)
        ax[cnt].title.set_text("Topic # " + str(k))
        cnt+=1
        
        if cnt==3:
            cnt=0
            fig, ax = plt.subplots(1, 3, figsize=(15,4))
    
    
              

def create_wordcloud(model, topic_idx, fig, ax, save=False):
    '''Wrapper function that generates individual wordclouds from topics in a tomotopy model
    
    ** Inputs **
    model: obj -> tomotopy trained model
    topic_idx: int -> topic index
    fig, ax: obj -> pyplot objects from subplots method
    save: bool -> If the user would like to save the images
    
    ** Returns **
    wordclouds as plots'''
    wcloud = WordCloud(background_color='white')
    
    topic_freqs = dict(model.get_topic_words(topic_idx))
    
    img = wcloud.generate_from_frequencies(topic_freqs)
    
    ax.imshow(img, interpolation='bilinear')
    ax.axis('off')
    
    if save:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        img_name = "wcloud_topic_" + str(topic_idx) +'.png'
        plt.savefig(''.join(['imgs/',img_name]), bbox_inches=extent.expanded(1.1, 1.2))
        
