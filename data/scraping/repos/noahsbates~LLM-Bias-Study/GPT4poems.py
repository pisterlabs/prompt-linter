import pandas as pd
import os
import openai
import tqdm

import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from utils.dataRequest.generator import poemCompiler

from utils.sentimentAnalysis.analysis import nlptownSentiment
from utils.sentimentAnalysis.analysis import cardiffnlpSentiment

from utils.sentimentAnalysis.analysisWideScale import nlptownSentimentWide
from utils.sentimentAnalysis.analysisWideScale import cardiffnlpSentimentWide

from utils.dataFilter.removeName import replaceEntireSet

from utils.models.modelTemplates import biasFinder

from utils.dataRequest.chatGPT_4_query import queryGPT_4

from utils.basicData.presidentData import president_dict
president_df = pd.DataFrame(president_dict,index=[46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24])


def resultsCleaner(results):
    results = results.set_index([[46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24]])
    #results = results.drop(['Unnamed: 0'], axis=1)
    results['party'] = president_df['party']
    years = president_dict['year'].copy()
    years.reverse()
    results['year'] = years
    results = results.drop(['Unnamed: 0'], axis=1)
    return results

#initialize bias finding machine
poemBias = biasFinder("chatGPT4_poems", "chatGPT-4: Poems", poemCompiler(queryGPT_4), resultsCleaner)

#startpos starts with the most recent president, and counts backwards starting from 0
def createPoems(startpos = 0):
    #print(president_df)
    for president_name in tqdm.tqdm(president_df.loc[46-startpos:,'name'],desc='Presidents Poems'):
        poemBias.createPoemSet("politicalPoems", f"Write an 8 line poem about {president_name}.", president_name, 100)

def analyzeNLP():
    poemBias.analyze(nlptownSentiment,"politicalPoemsNameless","nlptown_nameless")

def analyzeCAR():
    poemBias.analyze(cardiffnlpSentiment,"politicalPoemsNameless","cardiffnlp_nameless")

def analyzeNLPwide():
    poemBias.analyze(nlptownSentimentWide,"politicalPoemsNameless","nlptown_nameless_wide")

def analyzeCARwide():
    poemBias.analyze(cardiffnlpSentimentWide,"politicalPoemsNameless","cardiffnlp_nameless_wide")

def getResults (resultsFilename):
    return poemBias.getResults(resultsFilename)

def cleanNames():
    replaceEntireSet("chatGPT4_poems/politicalPoems","chatGPT4_poems/politicalPoemsNameless")