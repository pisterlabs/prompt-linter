from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary
from gensim import corpora
import numpy as np
import os
import sys
import time
import argparse
import LDA

'''
Best coherence for umass is typically the minimum. 
Best coherence for c_v is typically the maximum.
Umass is faster than c_v, but in my experience c_v gives better scores for optimal number of topics. 
This is not a hard decision rule. It depends on the use case. 

'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', help="a pickle file", type=str,  default = 'Step2_clean_data.pkl')
    parser.add_argument('--top_rg', help="Define the range of topic number [5,105] , min number and max num", nargs="+", default = [5,105])
    parser.add_argument('--pace', help="Define the pace of iteration", nargs="+",type=int, default = 5)
    parser.add_argument('--seed', help="Define your random state ",type=int, default = 12345)
    args = parser.parse_args()

    data = LDA.Load_data(args.data)
    print('start complexity ')
    for top_num in range(args.top_rg[0],args.top_rg[1],args.pace):
        start  = time.time()
        path = LDA.create_path(top_num,args.data)
        dictionary,corpus,ldamodel = LDA.Train_LDA(path,data,top_num,args.seed)
        print("123")
        model_Umass = CoherenceModel(model=ldamodel, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        model_c_v = CoherenceModel(model=ldamodel, texts=data, dictionary=dictionary,  coherence='c_v')
        u = model_Umass.get_coherence()
        cv = model_c_v.get_coherence()
        sys.stdout.write(f'Topic number {top_num}  is finished and use time {time.time() -start}')
        sys.stdout.flush()
        # with open("u_mass.txt", "a") as f:
        #     f.write(f"{u}\n")
        # with open("c_v.txt", "a") as f:
        #     f.write(f"{cv}\n")
        break
