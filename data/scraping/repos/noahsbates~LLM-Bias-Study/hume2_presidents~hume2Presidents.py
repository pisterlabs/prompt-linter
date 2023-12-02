import pandas as pd
import os
import openai
import tqdm
from transformers import pipeline

import numpy as np

#from utils.sentimentAnalysis.analysisWideScale import cardiffnlpSentimentWideIndividual

from utils.basicData.presidentData import president_dict
president_df = pd.DataFrame(president_dict,index=[46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24])

from utils.basicData.sentimentQuestions import positive50, neutral50, negative50

allSentimentQuestions = positive50 + neutral50 + negative50

import asyncio

from utils.dataRequest.config import API_KEYS
from hume import HumeStreamClient
from hume.models.config import LanguageConfig


#asyncio.run(analyzer.gather_sentiment(samples, "sentiment_results", api_key))

from utils.models.modelTemplates import sentimentLLMFinderHUME
hume2BiasFinder = sentimentLLMFinderHUME("hume2_presidents", allSentimentQuestions, resultsCleaner = None, api_key = API_KEYS['hume2'])

def findBias():
    asyncio.run(hume2BiasFinder.analyze(president_df["name"], "presidentNamesComplete"))

def getResults(fileName):
    return hume2BiasFinder.getResults(fileName)