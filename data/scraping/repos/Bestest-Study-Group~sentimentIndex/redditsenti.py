from math import floor
import cohere
from cohere.classify import Example
from dotenv import load_dotenv
import os
import praw
from datetime import datetime

load_dotenv()
COHERE = os.getenv('COHERE')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET') 

co = cohere.Client(COHERE)

def get_reddit_posts():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent='my user agent'
    )

    subreddit = reddit.subreddit('stocks')

    # submissions = subreddit.search(stock, sort='relevance', time_filter='month')
    submissions = subreddit.top(time_filter='week', limit=32)

    return submissions


def classify_submissions(submissions):
    fmt = '%Y-%m-%d'
    titles = []
    dates = []
    i = 0
    for submission in submissions:
        titles.append(submission.title)
        # dates.append(datetime.fromtimestamp(submission.created_utc).strftime(fmt))
        dates.append(submission.created_utc)
        i = i + 1
        if (i > 31):
            break

    classifications = co.classify(
        model='medium',
        taskDescription='Classify these as positive, negative',
        outputIndicator='Classify this stock',
        inputs=titles,
        examples=[
        Example("More Room For Growth In Pioneer Energy Stock?", "positive"),
        Example("After Dismal Performance Last Month, L'Oreal Stock Looks Set To Rebound", "positive"),
        Example("The stock market is close to finding its bottom as corporate share buybacks surge to record highs, JPMorgan says", "positive"),
        Example("How Do You Stay Confident in a Market Crash?", "negative"),
        Example("Here's 1 of the Biggest Problems With Airbnb Stock", "negative"),
        Example("GameStop Unveils Crypto and NFT Wallet, Shares up 3%", "positive"),
        Example("Should You Buy Stocks With An Impending Bear Market And Possible Recession?", "negative"),
        Example("Costco Q3 Earnings Preview: Don't Fall With It Any Longer (NASDAQ:COST)", "negative"),
        Example("Bear Market Has Only Just Begun", "negative"),
        Example("Photronics stock gains on guiding FQ3 above consensus after FQ2 beat (NASDAQ:PLAB)", "positive"),
        Example("Texas Instruments Stock: Playing The Long Game; Buy Now (NASDAQ:TXN)", "positive"),
        Example("U.S.-NEW YORK-STOCK MARKET-RISE", "positive"),
        Example("Chart Check: Record high in sight! This stock from agrochemical space is a good buy on dips bet", "positive"),
        Example("MSCI Inc. stock rises Wednesday, still underperforms market", "negative"),
        Example("DraftKings Inc. stock rises Wednesday, outperforms market", "positive"),
        Example("Willis Towers Watson PLC stock falls Tuesday, still outperforms market", "positive"),
        Example("ONEOK Inc. stock rises Tuesday, outperforms market", "positive"),
        Example("Marathon Oil Corp. stock falls Tuesday, still outperforms market", "positive"),
        Example("Intuitive Surgical Inc. stock falls Tuesday, underperforms market", "negative"),
        Example("Kohl's Corp. stock falls Monday, underperforms market", "negative"),
        Example("Intuit Inc. stock rises Monday, still underperforms market", "negative"),
        Example("Dow Inc. stock falls Monday, underperforms market", "negative"),
        Example("Walgreens Boots Alliance Inc. stock rises Thursday, still underperforms market", "negative"),
        Example("Waste Management Inc. stock rises Thursday, still underperforms market", "negative"),
        Example("Teleflex Inc. stock rises Thursday, still underperforms market", "negative"),
        Example("Public Storage stock rises Thursday, still underperforms market", "negative"),
        Example("Kohl's Corp. stock rises Thursday, outperforms market", "positive"),
        Example("Johnson Controls International PLC stock rises Thursday, outperforms market", "positive"),
        Example("Regency Centers Corp. stock rises Friday, outperforms market", "positive"),
        Example("Snap-On Inc. stock rises Friday, still underperforms market", "negative"),
        Example("Cooper Cos. stock rises Friday, still underperforms market", "negative"),
        Example("Unum Group stock rises Wednesday, still underperforms market", "negative"),
        Example("United Rentals Inc. stock rises Wednesday, outperforms market", "positive"),
        Example("Target Corp. stock outperforms market on strong trading day", "positive"),
        Example("Snap Inc. stock rises Wednesday, outperforms market", "positive"),
        Example("Paramount Global Cl B stock outperforms market on strong trading day", "positive"),
        Example("Live Nation Entertainment Inc. stock rises Wednesday, outperforms market", "positive"),
        Example("International Flavors & Fragrances Inc. stock rises Wednesday, still underperforms market", "negative"),
        Example('The Nasdaq fell 2.5% today, while TSLA fell 8%', 'negative')
    ])

    output = []
    i = 0 
    for cl in classifications.classifications:
        # print("TITLE: {}".format(cl.input))
        # print("SENTIMENT: {}".format(cl.prediction))
        # print("POSITIVE CONFIDENCE: {}".format(cl.confidence[0].confidence))
        # print("NEGATIVE CONFIDENCE: {}".format(cl.confidence[1].confidence))
        # print("DATE: {}".format(dates[i]))
        output.append({
            'title': cl.input,
            'sentiment': cl.prediction,
            'date': dates[i],
            'confidence': {
                'positive': cl.confidence[0].confidence,
                'negative': cl.confidence[1].confidence
            }
        })
            # temp[]
        # output.append(IndividualClassification(cl.input, cl.prediction, dates[i], cl.confidence[0].confidence, cl.confidence[1].confidence))
        i = i + 1
    
    temp_data = {}

    for out in output:
        today = datetime.now()
        temp = datetime.fromtimestamp(out['date'])
        days = floor((today - temp).total_seconds() / (60*60*24))
        
        if (days in temp_data):
            if (out['sentiment'] == 'negative'):
                temp_data[days].append(-1 * out['confidence']['negative'])
            else:
                temp_data[days].append(out['confidence']['positive'])
        else:
            if (out['sentiment'] == 'negative'):
                temp_data[days] = [(-1 * out['confidence']['negative'])]
            else:
                temp_data[days] = [(out['confidence']['positive'])]
    
    final = [None]*30

    for td in temp_data:
        sum = 0
        for t in temp_data[td]:
            sum += t
        
        final[td-1] = sum / len(temp_data[td])
        

        
    return final
    # return('The confidence levels of the labels are: {}'.format(
    #         classifications.classifications))

def run():
    posts = get_reddit_posts()
    classifications = classify_submissions(posts)
    # for cl in classifications:
    #     print("{}: {}".format(cl.title, cl.sentiment))
    return classifications