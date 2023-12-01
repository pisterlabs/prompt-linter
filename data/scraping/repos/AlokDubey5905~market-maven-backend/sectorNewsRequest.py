import models.sector_news as sector_news
import models.sentiment_analysis as sentiment_analysis
import pandas as pd
import models.get_cumulative_sentiment as get_cumulative_sentiment



def sector(name):
    # name="WMT"
    # function to get the stock news about the specific company
    print('--------------------------------Sector News--------------------------------------')
    df=sector_news.sectorNews(name)
    # print(df)

    # function to get the aggregated news about the same company
    # print('--------------------------------Aggregating News----------------------------------')
    # agg_df=aggregating_news.aggregate_rows(df)
    # agg_df.to_csv('data/sector/aggregated_sector_news.csv')

    # to get the stock data about the same company
    # print('--------------------------------Stock Data----------------------------------------')
    # stockDf=stockData.stockData(name)
    # stockDf.to_csv('data/sector/sector_stock_price.csv')

    # get the sentiment of the news...
    print('--------------------------------sector News Sentiment------------------------------------')

    # get the sentiment from textBlob model
    # df=pd.read_csv('data/sector/sector_news.csv')
    # df['sentiment']=df['headline'].apply(sentiment_analysis.textBlob)
    # df.to_csv('data/sector/textBlob_sector_news_sentiment.csv')

    # get sentiments from flair model
    # df=pd.read_csv('data/company_news.csv')
    # df['sentiment']=df['headline'].apply(sentiment_analysis.flairpred)
    # df.to_csv('data/company/flair_company_news_sentiment.csv')

    # get sentiments from vaderSentiment model
    df=pd.read_csv('data/sector/sector_news.csv')
    df['sentiment']=df['headline'].apply(sentiment_analysis.vaderPredict)
    df.to_csv('data/sector/vader_sector_news_sentiment.csv')

    # get sentiment from cohere model
    # df=pd.read_csv('data/sector/sector_news.csv')
    # cohere_df=sentiment_analysis.cohere_sentiment(df)
    # cohere_df.to_csv('data/sector/cohere_sector_news_sentiment.csv')

    # get sentiment from finBert model
    # df=pd.read_csv('data/sector/sector_news.csv')
    # finbert_df=sentiment_analysis.finModel(df['headline'].values.tolist(),df)
    # finbert_df.to_csv('data/sector/fin_sector_news_sentiment.csv')

    # get the cumulative sentiment for all the news in a day.
    df=pd.read_csv('data/sector/vader_sector_news_sentiment.csv')
    sector_dated_sentiment=get_cumulative_sentiment.cumulative_sentiment(df)
    sector_dated_sentiment.to_csv('data/sector/sector_dated_sentiment.csv')

