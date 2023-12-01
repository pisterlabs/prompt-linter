import pandas as pd
#import numpy as np
import pickle
import time
#import gc

from gensim.models.coherencemodel import CoherenceModel


# data needed for coherence calculation

# import entire dataset
f = open('../../../data/prd/Paper/coherence_vars.sav', 'rb')
[id2word, docs] = pickle.load(f)
f.close()

# import topics

df_topics = pd.read_pickle("./results/NMF/nmf_topics4-6.pkl")
nrow, ncol = df_topics.shape


print("data ingested--------------------------", flush = True)

# corpus - word frequency in docs - not needed for coherence calculation
# id2word - dictionary
# docs - df["final_tokens"]

# calculate coherence

n_topics = list(range(5,51,5))
batch = 4

col_names = [f"iteration {i+batch}" for i in range(ncol)]
co_val = pd.DataFrame(index = n_topics, columns = col_names)
co_t = pd.DataFrame(index = n_topics, columns = col_names)

for j in range(ncol):
    
    print(f'Iteration {j}', flush = True)
    
    coherence_values = []
    coherence_time = []
    
    for i in range(nrow): 
            
        # calculate coherence
        t1 = time.time()
        cm = CoherenceModel(topics=df_topics.iloc[i,j], dictionary=id2word, texts=docs, coherence='c_v', processes=15) 
        coherence_values.append(cm.get_coherence())
        t2 = time.time()
        coherence_time.append(t2-t1)
        print(f"  Coherence time: {t2-t1}", flush=True)
        
        # output completion message
        print('Number of topics =', df_topics.index[i], "complete.", flush = True)    
    
    # save results
    co_val[f"iteration {j+batch}"] = coherence_values
    co_t[f"iteration {j+batch}"] = coherence_time
    
       
        
# save results 

co_val.to_pickle("./results/NMF/co_nmf_val4-6.pkl")
co_t.to_pickle("./results/NMF/co_nmf_t4-6.pkl")
