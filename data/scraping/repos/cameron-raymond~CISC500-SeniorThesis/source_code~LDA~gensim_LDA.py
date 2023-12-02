import sys
import os
import tqdm
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim.corpora as corpora
from matplotlib import cm
from gensim.test.utils import datapath
from mpl_toolkits.mplot3d import Axes3D
from gensim.models import CoherenceModel

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def create_bow(data):
    # Create Dictionary
    word_dict = None
    if os.path.exists('./gensimmodel/worddict.txtdic'):
        print("--- using existing word dict ---")
        word_dict = corpora.Dictionary.load('./gensimmodel/worddict.txtdic')
    else:
        word_dict = corpora.Dictionary(data)  # Create Corpus
        word_dict.save('./gensimmodel/worddict.txtdic')
    # Term Document Frequency
    corpus = [word_dict.doc2bow(text) for text in data]  # View
    return corpus, word_dict

def compute_coherence_values(corpus, text_data, dictionary, k, a, b):
    """
    Computer the c_v coherence score for an arbitrary LDA model.

    For more info on c_v coherence see:  `M. Röder, A. Both, and A. Hinneburg: Exploring the Space of Topic Coherence Measures. 2015.`
    Parameters
    ----------
    :param corpus: the text to be modelled (a list of vectors).
    
    :param text_data: the actual text as a list of list
    
    :param dictionary: a dictionary coresponding that maps elements of the corpus to words.
    
    :param k: the number of topics
    
    :param a: Alpha, document-topic density
    
    :param b: Beta, topic-word density
    """
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b,
                                           per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=text_data, dictionary=dictionary, coherence='c_v')
    return coherence_model_lda.get_coherence()

def hyper_parameter_tuning(corpus, word_dict, text_data,min_topics=4,max_topics=8):
    """
    Iterates through an arbitrary number of hyperparameter combinaitions for a given corpus. 
    Measures LDA coherence
    Parameters
    ----------
    :param corpus: the text to be modelled (a list of vectors).
    
    :param word_dict: a dictionary coresponding that maps elements of the corpus to words.
    
    :param text_data: the actual text as a list of list

    :param min_topics: `optional` the minimum number of latent topics.

    :param max_topics: `optional` the maximum number of latent topics.
    """
    min_topics = min_topics
    max_topics = max_topics
    topics_range = range(min_topics, max_topics)
    step = 0.05
    # Alpha parameter
    alpha = list(np.arange(0.01, 1, step))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    # Beta parameter
    beta = list(np.arange(0.01, 1, step))
    beta.append('symmetric')
    model_results = {'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                     }
    num_combinations = len(topics_range)*len(alpha)*len(beta)
    pbar = tqdm.tqdm(total=num_combinations)
    # iterate through number of topics, different alpha values, and different beta values
    for num_topics in topics_range:
        for a in alpha:
            for b in beta:
                # get the coherence score for the given parameters
                cv = compute_coherence_values(
                    corpus=corpus, text_data=text_data, dictionary=word_dict, k=num_topics, a=a, b=b)
                model_results['Topics'].append(num_topics)
                model_results['Alpha'].append(a)
                model_results['Beta'].append(b)
                model_results['Coherence'].append(cv)
                pbar.update(1)
    pbar.close()
    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    best_val = np.argmax(model_results["Coherence"])
    print("Best c_v val: {} (alpha: {}, beta: {}, topics: {})".format(model_results['Coherence'][best_val],model_results['Alpha'][best_val], model_results['Beta'][best_val],model_results['Topics'][best_val]))
    return model_results['Coherence'][best_val],model_results['Alpha'][best_val], model_results['Beta'][best_val],model_results['Topics'][best_val]

def vis_coherence_surface(file_path,topics=10):
    """
        Visualizes the various hyper-parameters and their coherence score for a set number of topics.
    """
    ticks = lambda x : -0.6 if x == "asymmetric" else -0.4 if x == "symmetric" else x
    data = pd.read_csv(file_path)
    data = data[data["Topics"]==topics]
    x = data["Alpha"].apply(ticks).astype('float64')
    y = data["Beta"].apply(ticks).astype('float64')
    z = data["Coherence"].astype('float64')
    fig = plt.figure()
    ax = Axes3D(fig)
    # pylint: disable=no-member
    surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.05)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Coherence (c_v)')
    a = list(np.round(np.arange(-0.6, 1.2, 0.2),decimals=1))
    a[1] = "asymmetric"
    a[2] = "symmetric"
    ax.set_xticklabels(a)
    ax.set_yticklabels(a)
    plt.title("Alpha-Beta Hyperparameter Sweep (k={})".format(topics))
    plt.savefig('Coherence_Surface_k={}.png'.format(topics))

def return_hyperparams(corpus,word_dict,text_data,use_existing=True,**kwargs):
    """
        Returns the optimal hyperparameters. Done by sorting saved hyperparams or performing a new hyperparameter sweep.
    """
    to_float = lambda x : x if x=="symmetric" or x=="asymmetric" else float(x)
    exists = os.path.exists("lda_tuning_results.csv")
    params = None
    if not use_existing or not exists:
        print("--- starting hyperparameter tuning ---")
        coherence,alpha,beta,num_topics = hyper_parameter_tuning(corpus, word_dict, text_data,**kwargs)
        return coherence,alpha,beta,num_topics
    params = pd.read_csv("lda_tuning_results.csv")
    params["Alpha"],params["Beta"] = params["Alpha"].apply(to_float),params["Beta"].apply(to_float)
    best_val = params["Coherence"].idxmax()
    return params["Coherence"].loc[best_val],params["Alpha"].loc[best_val],params["Beta"].loc[best_val],params["Topics"].loc[best_val]

def predict(new_doc,lda_model,word_dict):
    try:
        new_doc = new_doc.split()
        BoW = word_dict.doc2bow(new_doc)
        doc_topics, _, _ = lda_model.get_document_topics(BoW, per_word_topics=True)
        return sorted(doc_topics,key=lambda x: x[1], reverse=True)[0][0]
    except:
        # Some rows may have null clean text (example: every token in the tweet is <3 character long). If that's the case return -1 (we want to discard these)
        return -1

def add_cluster(username,lda_model,word_dict):
    """
    Addeds the LDA clusters to a users twitter timeline data frame.
    Parameters
    ----------
    :param username: The twitter handle of the user being modelled.
    
    :param lda_model: Some gensim LDA model.
    
    :param word_dict: a dictionary coresponding that maps elements of the corpus to words.
    """
    file_path = "../data/{}_data.csv".format(username)
    timeline_df = pd.read_csv(file_path)
    timeline_df["lda_cluster"] = timeline_df["clean_text"].apply(lambda x : predict(x,lda_model,word_dict))
    csvFile = open(file_path, 'w' ,encoding='utf-8')
    timeline_df.to_csv(csvFile, mode='w', index=False, encoding="utf-8")

if __name__ == "__main__":
    # Put all of the party leaders into one data frame
    usernames = sys.argv[1:]
    frames = []
    for username in usernames:
        print(username)
        file_path = "../data/{}_data.csv".format(username)
        timeline_df = pd.read_csv(file_path)
        print("Number of Tweets for {} is {}".format(username, len(timeline_df)))
        frames.append(timeline_df)
    # The sample(frac=1) shuffles the rows
    text_data = pd.concat(frames,sort=False)["clean_text"].sample(frac=1).values.astype('U')
    #Convert each tweet into a list of tokens
    text_data = [sent.split() for sent in text_data]
    # Build the bigram models
    print("--- finding bigrams ---")
    bigram = gensim.models.Phrases(text_data, min_count=8, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # creates bigrams of words that appear frequently together "gun control" -> "gun_control"
    text_data = make_bigrams(text_data, bigram_mod)
    print("--- creating BoW model ---")
    corpus, word_dict = create_bow(text_data)
    print("--- returning hyperparameters ---")
    min_topics = 4
    max_topics = 7
    coherence,alpha,beta,num_topics = return_hyperparams(corpus, word_dict, text_data,use_existing=True)
    # coherence,alpha,beta,num_topics = return_hyperparams(corpus, word_dict, text_data,use_existing=False,min_topics=min_topics,max_topics=max_topics)
    # Build LDA model
    print("--- Building model with coherence {:.3f} (Alpha: {:.2f}, Beta: {:0.2f}, Num Topics: {}) ---".format(coherence,alpha,beta,num_topics))
    lda_model = gensim.models.LdaMulticore(corpus=corpus,id2word=word_dict,num_topics=num_topics,alpha=alpha,eta=beta,random_state=100,chunksize=100,passes=10,per_word_topics=True)
    print("--- Updating {} Users Tweet Clusters ---".format(len(usernames)))
    pbar = tqdm.tqdm(total=len(usernames))
    for username in usernames:
        add_cluster(username,lda_model,word_dict)
        pbar.update(1)
    pbar.close()
    # for i in range(min_topics,max_topics):
    #     vis_coherence_surface("lda_tuning_results.csv",topics=i)
    # for idx, topic in lda_model.print_topics(-1):
    #     print('Topic: {} \nWords: {}'.format(idx, topic))
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=text_data, dictionary=word_dict, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # print('\nCoherence Score: {}'.format(coherence_lda))
    # temp_file = datapath("./gensimmodel/gensim_lda")
    # lda_model.save(temp_file)

