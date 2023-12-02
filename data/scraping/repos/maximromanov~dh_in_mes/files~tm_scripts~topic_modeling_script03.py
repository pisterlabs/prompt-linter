# the script the coherence score test

import io
import pandas

# Gensim
import gensim
import gensim.corpora
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from ast import literal_eval # for loading columns with lists

###############################################################################################
# VARIABLES ###################################################################################
###############################################################################################

seedValue = 2023 # this one is needed for reproducability
minTopic   = 2
maxTopic   = 60
fileSuffix = "_years_1860_1864_topics_Min%02d_Max%02d" % (minTopic, maxTopic) 

###############################################################################################
def main():
    print("""
    ######################################################
    Running coherence score test for %d-%d topics
    ::: %s
    random_state value: %d
    ######################################################
    """ % (minTopic, maxTopic, fileSuffix, seedValue))

    # LOAD CORPUS (WE CAN EDIT THIS LIST IN ORDER TO REDUCE THE AMOUNT OF DATA THAT WE ARE LOADING)
    print("="*80)
    print("Loading the corpus")
    dispatchSubfolder = "./Dispatch_Processed_TSV/"
    dispatchFiles = ["Dispatch_1860_tmReady.tsv",  # incomplete
                    "Dispatch_1861_tmReady.tsv",  # The War starts of April 12, 1861
                    "Dispatch_1862_tmReady.tsv",
                    "Dispatch_1863_tmReady.tsv",
                    "Dispatch_1864_tmReady.tsv",
                    #"Dispatch_1865_tmReady.tsv",  # incomplete - The War ends on May 9, 1865
                    ]

    df = pandas.DataFrame()
    for f in dispatchFiles:
        print("\t", f)
        dfTemp = pandas.read_csv(dispatchSubfolder + f, sep="\t", header=0, converters={'textDataLists': literal_eval})
        df = df.append(dfTemp)

    dispatch_light = df
    # drop=True -- use it to avoid creating a new column with the old index values
    dispatch_light = dispatch_light.reset_index(drop=True) 
    dispatch_light["month"] = pandas.to_datetime(dispatch_light["month"], format="%Y-%m")
    dispatch_light["date"] = pandas.to_datetime(dispatch_light["date"], format="%Y-%m-%d")
    print("="*80)

    # PREPARE CORPUS FOR TOPIC MODELING - generate gensim objects: dictionary, corpus, term frequencies (bag of words) 
    print("\tpreparing the corpus")
    dictionary = gensim.corpora.Dictionary(dispatch_light["textDataLists"])
    texts = dispatch_light["textDataLists"]
    corpus = [dictionary.doc2bow(text) for text in texts] # bow == bag of words

    # run optimal number test
    print("\trunning optimal number test...")
    print("="*80)
    optimalTopicsNumber = ["topics\tscore"]

    for num in range(minTopic, maxTopic+1, 1):
        lda_model_temp = gensim.models.LdaModel(corpus=corpus,id2word=dictionary, num_topics=num, random_state=seedValue)
        coherence_model_lda = CoherenceModel(model=lda_model_temp,texts=dispatch_light["textDataLists"],
                                        dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        optimalTopicsNumber.append("%d\t%f" % (num, coherence_lda))
        print('\t\tCoherence Score for %02d topics: ' % num, coherence_lda)
        
    print("-"*50)

    # save results
    print("\tSaving results...")
    optimalTopicsNumber = "\n".join(optimalTopicsNumber)
    with open(dispatchSubfolder + "optimal_topic_number_%s.tsv" % fileSuffix, "w", encoding="utf8") as f9:
        f9.write(optimalTopicsNumber)
    print("-"*50)

    # graph results
    scoresData = io.StringIO(optimalTopicsNumber)
    scoresDF = pandas.read_csv(scoresData, sep="\t", header=0)

    plt.rcParams["figure.figsize"] = (20, 9)
    plt.stem(scoresDF['topics'], scoresDF['score'])
    plt.plot([minTopic, maxTopic], [0.55, 0.55], color="red", linestyle="--")

    plt.ylabel("coherence score")
    plt.xlabel("number of topics")
    plt.title("Coherence Score Test for TM of the Dispatch")
    plt.gca().yaxis.grid(linestyle=':')

    plt.savefig(dispatchSubfolder + "optimal_topic_number_%s.pdf" % fileSuffix, dpi=150)
    #plt.show()

###############################################################################################
if __name__ == "__main__":
    main()
###############################################################################################