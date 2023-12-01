import cohere
from cohere.classify import Example, Classification
from cohere.response import CohereObject
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv('COHERE_API_KEY','RMppgVMUjgiKZRWSIwjmmfbOwRa9YEhi1B15oxQ2'))

examples=[
Example("You're an idiot", "negative"),
  Example("I hate you", "negative"),
  Example("You are stupid", "negative"),
  Example("I will never forgive you", "negative"),
  Example("What a loser", "negative"),
  Example("You're an idiot", "negative"),
  Example("I hate you", "negative"),
  Example("You are stupid", "negative"),
  Example("I will never forgive you", "negative"),
  Example("What a loser", "negative"),
  Example("I am feeling great today!", "positive"), 
  Example("I just got a promotion!", "positive"), 
  Example("I love spending time with my family", "positive"), 
  Example("I am so excited for tomorrow", "positive"), 
  Example("I am feeling content", "positive"), 
  Example("I had a wonderful time at the park", "positive"), 
  Example("I am looking forward to my vacation", "positive"), 
  Example("This meal was delicious", "positive"), 
  Example("I am so proud of my accomplishments", "positive"),
  Example("The weather is mild", "neutral"),
  Example("I ran for an hour", "neutral"),
  Example("The movie was average", "neutral"),
  Example("The store was closed", "neutral"),
  Example("I took the bus home", "neutral"),
  Example("The food was okay", "neutral"),
  Example("It was an average day", "neutral"),
]

def classify_sentiment(text):
    response = co.classify(
        model='large',
        inputs=[text],
        examples=examples,
    )
    # print(response.classifications[0])
    return response.classifications[0]

# sample output The confidence levels of the labels are: [{'negative': 0.9999999999999999, 'positive': 1.1102230246251565e-16, 'neutral': 1.1102230246251565e-16}]
def get_sentiment_percentage(text):
    response = classify_sentiment(text)
    negative_score = response.labels['negative'].confidence
    positive_score = response.labels['positive'].confidence
    neutral_score = response.labels['neutral'].confidence
    return [negative_score, positive_score, neutral_score]
def get_sentiment_result(text):
    response = classify_sentiment(text)
    return response.prediction
