import models.companyNews as companyNews
import models.stockData as stockData
import models.sentiment_analysis as sentiment_analysis
import pandas as pd
import models.get_cumulative_sentiment as get_cumulative_sentiment


def company(name):
    # company name
    # name="WMT"
    # function to get the stock news about the specific company
    print('--------------------------------Company News--------------------------------------')
    df = companyNews.companyNews(name)
    # print(df)

    # function to get the aggregated news about the same company
    # print('--------------------------------Aggregating News----------------------------------')
    # agg_df=aggregating_news.aggregate_rows(df)
    # agg_df.to_csv('data/company/aggregated_company_news.csv')

    # to get the stock data about the same company
    print('--------------------------------Stock Data----------------------------------------')
    stock_symbol = companyNews.find_stock_symbol(name)
    stockDf = stockData.stockdata(stock_symbol, name)
    stockDf.to_csv('data/company/company_stock_prices.csv')

    # get the sentiment of the news...
    print('--------------------------------company News Sentiment------------------------------------')

    # get the sentiment from textBlob model
    # df=pd.read_csv('data/company/company_news.csv')
    # df['sentiment']=df['headline'].apply(sentiment_analysis.textBlob)
    # df.to_csv('data/company/textBlob_company_news_sentiment.csv')

    # get sentiments from flair model
    # df=pd.read_csv('data/company_news.csv')
    # df['sentiment']=df['headline'].apply(sentiment_analysis.flairpred)
    # df.to_csv('data/company/flair_company_news_sentiment.csv')

    # get sentiments from vaderSentiment model
    # df=pd.read_csv('data/company/company_news.csv')
    # df['sentiment']=df['headline'].apply(sentiment_analysis.vaderPredict)
    # df.to_csv('data/company/vader_company_news_sentiment.csv')

    # get sentiment from cohere model
    df = pd.read_csv('data/company/company_news.csv')
    cohere_df = sentiment_analysis.cohere_sentiment(df)
    cohere_df.drop("Unnamed: 0", axis=1, inplace=True)
    cohere_df.to_csv('data/company/cohere_company_news_sentiment.csv')

    # get sentiment from finBert model
    # df=pd.read_csv('data/company/company_news.csv')
    # finbert_df=sentiment_analysis.finModel(df['headline'].values.tolist(),df)
    # finbert_df.to_csv('data/company/fin_company_news_sentiment.csv')

    # get the cumulative sentiment for all the news in a day.
    # df=pd.read_csv('data/company/fin_company_news_sentiment.csv')
    # company_dated_sentiment=get_cumulative_sentiment.cumulative_sentiment(df)
    # company_dated_sentiment.to_csv('data/company/company_dated_sentiment.csv')

    company_dated_sentiment = get_cumulative_sentiment.cumulative_sentiment(
        cohere_df)
    company_dated_sentiment.to_csv('data/company/company_dated_sentiment.csv')


def compn_news(stock_name):
    print('--------------------------------Company News--------------------------------------')
    df = companyNews.companyNews(stock_name)
    print('--------------------------------company News Sentiment------------------------------------')
    cohere_df = sentiment_analysis.cohere_sentiment(df)
    return cohere_df
