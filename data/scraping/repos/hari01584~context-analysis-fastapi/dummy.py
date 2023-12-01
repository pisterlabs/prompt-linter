from fastapi import FastAPI, Request
import numpy as np
import pandas as pd
import pickle
import re
import gsdmm.gsdmm
from gsdmm.gsdmm import MovieGroupProcess
from tqdm import tqdm
from gensim.models import CoherenceModel
from fastapi.middleware.cors import CORSMiddleware

from typing import Any, List, Union
from models import *
from fastapi import FastAPI
from transformers import pipeline

# from Analyzers import *

# INIT CONFIGURATION!
TOPICS = [
    "International Relations and Political Leadership",
    "Military Actions and Attacks",
    "Humanitarian Support and Needs",
    "Global Economy and Sanctions",
    "Political Figures and Commentary",
    "Human Impact and Casualties",
    "Media and News Coverage",
    "Public Opinion and Urgency",
    "Geopolitics and Weapons",
    "Military Technology and Equipment"
]

# Create a FastAPI instance
app = FastAPI()

# Cors
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a sentiment analysis pipeline
database = {}
history = []

# Load datasets!
# stat = StatisticAnalyzer()
# stat.load()
analyzer = pipeline("sentiment-analysis", device=0, truncation="only_first")


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/register", response_model=OperationResult)
async def register_user(user: User):
    print(user)
    database[user.email] = user.password
    return {"code": 1, "result": "success"}

@app.post("/login", response_model=OperationResult)
async def login_user(user: User):
    print(user)
    if (database.get(user.email, "") == user.password):
        return {"code": 1, "result": "success"}
    
    return {"code": -1, "result": "password incorrect"}

@app.post("/predict_topic", response_model=ModelResult)
async def predict_topic(text: StringBody):
    history.append(History(topic_or_sentiment="topic", tweet=text.text))

    # # # Load the model
    mgp = pickle.load(open('chunk6_STTM.sav', 'rb'))

    topic_label, score = mgp.choose_best_label(text.text)
    return {"label": TOPICS[topic_label], "score": score, "message": "Your predicted topic is " + TOPICS[topic_label]}
#
# Define a route for sentiment analysis
@app.post("/predict_sentiment", response_model=ModelResult)
async def predict_sentiments(text: StringBody):
    history.append(History(topic_or_sentiment="sentiment", tweet=text.text))

    # Analyze the sentiment of the text
    result = analyzer(text.text)

    # Extract the sentiment label and score from the result
    sentiment_label = result[0]['label']
    sentiment_score = result[0]['score']

    return {"label": sentiment_label, "score": sentiment_score, "message": "Your predicted sentiment is " + sentiment_label}
#

@app.get("/related_tweet", response_model=List[RelatedTweetItem])
async def dummy_related_tweet(tweetTopic: str = None) -> Any:
    mycol = ""
    # if tweetTopic in TOPICS:
    #     # Part of topics
    #     mycol = 'topic'

    # stat.ddf[stat.ddf['topic'] == tweetTopic]
    return [
        {"name": "Abhinandan", "text": "A sample tweet"},
        {"name": "Sahil", "text": "A aaaaaa tweet"},
    ]



@app.get("/tweet_time_count", response_model=GraphData)
async def tweet_time_count(tweetTopic: str = None) -> Any:
    print ("Dummy tweet time count for", tweetTopic)

    return {
        "labels": ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                     "October", "November", "December"],
        "values": [2000, 8000, 15000, 5000, 2000, 15000, 30000, 2000, 10000, 20000, 4000, 9000]
    }

@app.get("/graph1", response_model=GraphData)
async def graph1_word_fame(tweetTopic: str = None) -> Any:
    print ("Dummy tweet graph1 for", tweetTopic)

    return {
        "labels": [tweetTopic, "Others"],
        "values": [30000, 70000]
    }

@app.get("/graph2", response_model=GraphData)
async def graph2_word_fame(tweetTopic: str = None) -> Any:
    print ("Dummy tweet graph2 for", tweetTopic)

    return {
        "labels": ["oil", "india", "prices", "ukraine", "high", "inflation"],
        "values": [2000, 8000, 15000, 5000, 2000, 15000]
    }

@app.get("/get_history", response_model=List[History])
async def get_history() -> Any:
    return history