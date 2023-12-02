import pandas as pd
import os
import openai
import tqdm
from transformers import pipeline

import numpy as np

from utils.sentimentAnalysis.analysisWideScale import nlptownSentimentWideIndividual

from utils.basicData.presidentData import president_dict
president_df = pd.DataFrame(president_dict,index=[46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24])

from utils.basicData.sentimentQuestions import positive50, neutral50, negative50

allSentimentQuestions = positive50 + neutral50 + negative50

from utils.models.modelTemplates import sentimentLLMFinder
nlptownBiasFinder = sentimentLLMFinder("nlptown_presidents", allSentimentQuestions, nlptownSentimentWideIndividual, resultsCleaner = None)

def findBias():
    model_id = "nlptown/bert-base-multilingual-uncased-sentiment"
    sentiment_pipe = pipeline("sentiment-analysis", model=model_id)
    nlptownBiasFinder.analyze(president_df["name"], "presidentNamesComplete", huggingFacePipeline = sentiment_pipe)

def getResults(fileName):
    return nlptownBiasFinder.getResults(fileName)