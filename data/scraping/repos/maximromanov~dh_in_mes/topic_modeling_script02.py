# the script runs topic modeling on the Dispatch data and saves several results files in the process
# 1) the LDA model; 2) topic "names" json; 3) topic-word probabilities for SNA; 4) ModeledDataLight
# (includes original IDs and topic distributions)

import io, json
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
numberOfTopics   = 40
fileSuffix = "_years_1860_1864_%02dtopics" % (numberOfTopics) 

###############################################################################################
def main():
    print("""
    ######################################################
    Running topic modeling for %d topics
    ::: %s
    random_state value: %d
    ######################################################
    """ % (numberOfTopics, fileSuffix, seedValue))

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

    # RUNNING TOPIC MODELING
    print("\trunning topic modeling")
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary,
                                   random_state=2023,
                                   #update_every=20, passes=100, alpha='auto',
                                   num_topics=numberOfTopics)

    lda_model.save(dispatchSubfolder + "LDA_model_%s.lda" % fileSuffix)
    # the generated model can be loaded like this:
    # lda_model = gensim.models.LdaModel.load(dispatchSubfolder + "LDA_model_%s.lda" % fileSuffix)

    # save results for later SNA
    print("\tsave SNA representation")
    topicsDataNW = lda_model.print_topics(num_topics = numberOfTopics, num_words=20)
    topicsTidy = []
    topicsDicQuick = {}

    for t in topicsDataNW:
        topicsDicQuick[t[0]] = t[1]
        words = t[1].split(" + ")
        for w in words:
            w = w.replace('"', "").replace("*", "\t")
            topicsTidy.append("%s\tT%02d\t%s" % (t[0], int(t[0])+1, w))

    topicsTidy = "\n".join(topicsTidy)
    with open(dispatchSubfolder + "LDA_model_%s_TIDY_for_SNA.tsv" % fileSuffix, "w", encoding="utf8") as f9:
        f9.write("topic\ttopicName\tscore\tterm\n" + topicsTidy)

    # save "names" of topics
    print("\tsave topic 'names'")
    topicTableCols = [] # empty table for topic values (technically, a list still)
    topicDic = {} # a dictionary with top words per topic

    for i in range(0, numberOfTopics, 1):
        tVal = "T%02d" % (i + 1)
        topicTableCols.append(tVal)
        
        topicVals  = lda_model.show_topic(i)
        topicWords = ", ".join([word for word, prob in topicVals])
        topicDic[tVal] = topicWords

    with open(dispatchSubfolder + "LDA_model_%s_TOPICS.json" % fileSuffix, "w") as f9:
        json.dump(topicDic, f9, indent=4, ensure_ascii=False)

    # aggregate all the data - this is the longest part
    print("\taggregate all the data")
    all_topics = lda_model.get_document_topics(corpus, per_word_topics=True)
    topicTableRows = [] # now we are feeding topic values into our empty table

    for doc_topics, word_topics, phi_values in all_topics:
        rawRow = [0] * numberOfTopics
        for t in doc_topics:
            rawRow[t[0]] = t[1]
        topicTableRows.append(rawRow)

    # We just need to convert it into a proper dataframe format:
    topicTable = pandas.DataFrame(topicTableRows, columns=topicTableCols)
    dispatch_light = dispatch_light.reset_index(drop=True)
    dispatch_light = dispatch_light[["id"]]

    # merge our initial data with topics --- this is the main table that we produce (we only keep IDs from the original table to save space)
    mergedTable = pandas.concat([dispatch_light, topicTable], axis=1, sort=False)
    mergedTable.to_csv(dispatchSubfolder + "LDA_model_%s_ModeledDataLight.tsv" % fileSuffix, sep="\t", index=False)
    print("DONE!")


###############################################################################################
if __name__ == "__main__":
    main()
###############################################################################################