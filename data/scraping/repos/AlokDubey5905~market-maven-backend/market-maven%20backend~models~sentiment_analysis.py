from textblob import TextBlob
# from flair.models import TextClassifier
# from flair.data import Sentence
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import pandas as pd
import cohere
from cohere.responses.classify import Example
from transformers import pipeline


# Define the get_sentiment function using the python's text blob
def textBlob(statement):
    if isinstance(statement, float):
        statement = ''  # Convert NaN or float values to empty string

    blob = TextBlob(statement)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment


# Define the get_sentiment function using the python's flair
# def flairpred(text):
#     sentence = Sentence(text)
#     # Load the Flair sentiment model
#     classifier = TextClassifier.load('en-sentiment')
#     classifier.predict(sentence)
#     if len(sentence.labels) > 0:
#         label = sentence.labels[0]
#         return label.value
#     else:
#         return 'UNKNOWN'


# function to get the sentiment using the python's vadersentiment
def vaderPredict(text):
    # Initialize the SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    score = analyzer.polarity_scores(text)
    compound_score = score['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# function to get the sentiment using the cohere model


def predict_sentiment(text_list):
    # Replace with your API key
    cohere_client = cohere.Client('VmDB7D9AzAA6FsHhJFnAf5Cuy03kdBUTOIQCnccP')
    response = cohere_client.classify(
        model='fcc99961-dfc4-4d0e-b98c-3c7d0e85a2f9-ft',
        inputs=text_list)
    predictions = [item.prediction for item in response.classifications]
    return predictions


def cohere_sentiment(dataframe):
    companyNewsDf = dataframe

    batch_size = 90
    sentiments = []
    for i in range(0, len(companyNewsDf), batch_size):
        batch_df = companyNewsDf.iloc[i:i+batch_size]
        batch_text_list = batch_df['headline'].values.tolist()
        batch_sentiments = predict_sentiment(batch_text_list)
        sentiments.extend(batch_sentiments)

    companyNewsDf['sentiment'] = sentiments
    companyNewsDf['sentiment'] = companyNewsDf['sentiment'].replace(
        {'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'})

    # companyNewsDf.to_csv('data/cohere_sentiment_company_news_sentiment.csv', index=False)
    return companyNewsDf

# function to get the sentiment usning the finBert


def finModel(dflist, df1):
    # Load the FinBERT sentiment analysis model
    pipe = pipeline("text-classification", model="ProsusAI/finbert")

    # Get the sentiment predictions for the list of summaries
    predictions = pipe(dflist)
    sentiments = [prediction['label'] for prediction in predictions]

    # Add the sentiments to the DataFrame
    df1['sentiment'] = sentiments
    return df1


# def sentiments(dataframe):

#     company_news = dataframe
#     company_news.dropna(inplace=True)

    # # get the sentiment using the textblob
    # company_news['sentiment'] = company_news['headline'].apply(textBlob)
    # company_news.to_csv('data/textBlob_company_news_sentiment.csv')

    # # get the sentiment using flair
    # company_news['sentiment'] = company_news['headline'].apply(flairpred)
    # company_news.to_csv('data/flairpred_company_news_sentiment.csv')

    # # get the sentiment using vaderSentiment
    # company_news['sentiment'] = company_news['headline'].apply(vaderPredict)
    # company_news.to_csv('data/vaderPredict_company_news_sentiment.csv')

    # get the setiment using cohere model
    # company_news['sentiment'] = company_news['headline'].apply(cohere_sentiment)
    # cohere_senti=cohere_sentiment(company_news)

    # get the sentiment using the finbert model
    # finbert_senti=finModel(company_news['summary'].values.tolist(), company_news)
    # finbert_senti.to_csv('data/finModel_company_news_sentiment.csv')

    # return the updated dataframe
    # return finbert_senti
