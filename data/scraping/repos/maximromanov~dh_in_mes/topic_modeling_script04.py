# the script runs several analyses of the generated tm data

# the following is to ignore deprecation warnings from pyLDA library; in general, not a good practice as we need to
# write our code in a way that would make it reusable in the future; DeprecationWarning inform us that some element
# of a library in use will stop being usable soonl the annoying thing is the warnings are still printed out...
import warnings
warnings.simplefilter("ignore", FutureWarning, lineno=0)
warnings.filterwarnings("ignore")

import io, json, yaml
import pandas

# Gensim
import gensim
import gensim.corpora
from gensim.models import CoherenceModel

# library for LDA vis
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import matplotlib.pyplot as plt

from ast import literal_eval # for loading columns with lists

###############################################################################################
# VARIABLES ###################################################################################
###############################################################################################

seedValue = 2023 # this one is needed for reproducability
numberOfTopics = 40
fileSuffix = "_years_1860_1864_%dtopics" % numberOfTopics
dispatchSubfolder = "./Dispatch_Processed_TSV/"

# files to load
ldaModelFile = "LDA_model__years_1860_1864_%dtopics.lda" % numberOfTopics
topicsNamesFile = "LDA_model__years_1860_1864_%dtopics_TOPICS.json" % numberOfTopics
topicsData = "LDA_model__years_1860_1864_%dtopics_ModeledDataLight.tsv" % numberOfTopics

###############################################################################################
def main(LDAvisVar=False):
    print("""
    ######################################################
    Running analyses based on %d topics
    ::: %s
    random_state value: %d
    ######################################################
    """ % (numberOfTopics, fileSuffix, seedValue))

    # LOAD MODELS and TOPIC NAMES
    print("\tLoading pregenerated topic model: %s" % ldaModelFile)
    ldaModel = gensim.models.LdaModel.load(dispatchSubfolder + ldaModelFile)

    print("\tLoading pregenerated topic names: %s" % topicsNamesFile)
    with open(dispatchSubfolder + topicsNamesFile) as jsonData:
        topicNames = json.load(jsonData)

    # LOAD CORPUS (WE CAN EDIT THIS LIST IN ORDER TO REDUCE THE AMOUNT OF DATA THAT WE ARE LOADING)
    print("="*80)
    print("\tLoading the corpus")
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
    df = df.reset_index(drop=True)
    #print(len(df))

    print("\tLoading the topics data: %s" % topicsData)
    dfTopics = pandas.read_csv(dispatchSubfolder + topicsData, sep="\t", header=0)
    #print(len(dfTopics))

    print("\tMerging tables...")
    dispatch_light = pandas.merge(df, dfTopics, how="left", on="id")
    #print(len(dispatch_light))

    # drop=True -- use it to avoid creating a new column with the old index values
    dispatch_light = dispatch_light.reset_index(drop=True) 
    dispatch_light["month"] = pandas.to_datetime(dispatch_light["month"], format="%Y-%m")
    dispatch_light["date"] = pandas.to_datetime(dispatch_light["date"], format="%Y-%m-%d")
    print("="*80)

    #print(dispatch_light)
    if LDAvisVar:
        print("\tGenerating topics browser (LDAvis)")
        dictionary = gensim.corpora.Dictionary(dispatch_light["textDataLists"])
        texts = dispatch_light["textDataLists"]
        corpus = [dictionary.doc2bow(text) for text in texts]

        modelVis = pyLDAvis.gensim_models.prepare(ldaModel, corpus, dictionary)
        pyLDAvis.save_html(modelVis, dispatchSubfolder + 'tmVis_%s.html' % fileSuffix)
        print("\t\tsaved into: " + 'tmVis_%s.html' % fileSuffix)
    else:
        print("\tTopics browser (LDAvis) was not generated")

    # Generating analyses
    print("\tVisualizing topics over time (grouped by months) - % of a topic per month")
    print("\t\tand: Aggregating most representative articles by topic (into TEXT files)")

    def graphFunc(dataframe, dfType, topic, topicName):
        plt.rcParams["figure.figsize"] = (20, 9)
        plt.plot(dataframe['month'], dataframe[k])
        plt.ylabel("topic frequencies (%s)" % dfType)
        plt.xlabel("dates of issues of the Dispatch")
        plt.title(topic + ": " + topicName)
        plt.gca().yaxis.grid(linestyle=':')
        plt.savefig(dispatchSubfolder + "TopicChronoGraph_%s_%s_%s.pdf" % (fileSuffix, topic, dfType), dpi=150)
        plt.savefig(dispatchSubfolder + "TopicChronoGraph_%s_%s_%s.png" % (fileSuffix, topic, dfType), dpi=150)
        plt.clf()   

    # This will find the average mean for each topic for each month
    topicSumMean = dispatch_light.groupby("month").mean().copy()
    topicSumMean["month"] = topicSumMean.index

    # This will find the sum for each topic for each month
    topicSumSum = dispatch_light.groupby("month").sum().copy()
    topicSumSum["month"] = topicSumSum.index

    # This will find the rel for each topic for each month
    topicSumRel = dispatch_light.groupby("month").sum().copy()
    topicSumRel["total"] = topicSumRel.sum(axis=1) # this calculates the 100% value per month
    for column in topicSumRel:
        topicSumRel[column + "_ABS"] = topicSumRel[column]
        topicSumRel[column] = topicSumRel[column] / topicSumRel["total"] * 100
        topicSumRel[column + "_REL"] = topicSumRel[column]
    topicSumRel["month"] = topicSumRel.index

    # HM, graphs of MEAN are extremely similar to graphs of RELATIVE...
    #topicSumRel.to_csv(dispatchSubfolder + "_TEST.tsv", sep="\t", index=False)
    #print(topicSumRel)

    for k,v in topicNames.items():
        graphFunc(topicSumMean, "mean", k, v) # MEAN GRAPH
        graphFunc(topicSumSum, "sum", k, v) # SUM GRAPH
        graphFunc(topicSumRel, "relative", k, v) # RELATIVE GRAPH

        temp = dispatch_light.sort_values(by=k, ascending=False)
        temp = temp[["id", "text", k]].head(100)
        temp.to_csv(dispatchSubfolder + "TopicSamples_%s_%s.tsv" % (fileSuffix, k), sep="\t", index=False)
        with open(dispatchSubfolder + "TopicSamples_%s_%s.yml" % (fileSuffix, k), "w") as f9:
            yaml.dump(temp.to_dict(orient="records"), f9)


###############################################################################################
if __name__ == "__main__":
    main()
    #main(LDAvisVar=False)
###############################################################################################