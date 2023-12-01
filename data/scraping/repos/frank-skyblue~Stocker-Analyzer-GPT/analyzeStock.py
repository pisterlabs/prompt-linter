import json
from bs4 import BeautifulSoup
import re
import os
import requests

import openai
import langchain
from langchain.llms import OpenAI
import yfinance as yf

import warnings
warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = "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"

# Initialize GPT 3.5
llm=OpenAI(temperature=0,
           model_name="gpt-3.5-turbo-16k-0613")

# Function refers to getting "Company's name" and "stock ticker" from the query
function=[
        {
        "name": "stock_ticker",
        "description": "This will get the stock ticker of the company",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker_symbol": {
                    "type": "string",
                    "description": "Stock symbol of the company given in query",
                },

                "company_name": {
                    "type": "string",
                    "description": "Name of the company given in query",
                }
            },
            "required": ["company_name","ticker_symbol"],
        },
    }
]

# Get the company name and ticker from the query
# Example: "What is the stock ticker for Apple?"
# Output: "Apple", "AAPL"
def get_company_name_and_ticker(query):
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0, # Make outcome relatively deterministic
            messages=[{
                "role":"user",
                "content":f"Given the request from user, what is the Company's name as well as the company's stock ticker?: {query}?"
            }],
            functions=function,
            function_call={"name": "stock_ticker"},
    )
    message = response["choices"][0]["message"]
    arguments = json.loads(message["function_call"]["arguments"])

    # Extract the company name and ticker from the response
    company_name = arguments["company_name"]
    company_ticker = arguments["ticker_symbol"]

    return company_name,company_ticker

# Get the stock price of the company
# Example: "What is the stock price for Apple?"
# Output: the stock prices of Apple over the past 5 days
def get_stock_price(ticker,history=5):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    df=df[["Close","Volume"]]
    df.index=[str(x).split()[0] for x in list(df.index)]
    df.index.rename("Date",inplace=True)
    df=df[-history:]

    return df.to_string()

# Get the balance sheet of the company
# Example: "What is the balance sheet for Apple?"
# Output: the balance sheet of Apple for the current year
def get_company_balance_sheet(ticker):
    company = yf.Ticker(ticker)
    balance_sheet = company.balance_sheet
    if balance_sheet.shape[1]>=3:
        balance_sheet=balance_sheet.iloc[:,:3]
    balance_sheet=balance_sheet.dropna(how="any")
    balance_sheet = balance_sheet.to_string()

    return balance_sheet

# NOT IN USE                                                                                                
######################################################################################################################

def get_query_url(search_term):
    if "news" not in search_term:
        search_term = search_term + " stock news"
    url = f"https://www.google.com/search?q={search_term}&gl=ca"
    url = re.sub(r"\s", "+", url)
    return url

def get_recent_stock_news(company_name):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36'
    }

    url = get_query_url(company_name)
    res = requests.get(url, headers=headers).text
    soup = BeautifulSoup(res, "html.parser")

    news_links = []

    for link in soup.find_all("a"):
        href = link.get("href")
        if href and "url?q=" in href:
            url = re.findall(r"url\?q=(.+)&sa", href)[0]
            if "google" not in url:
                news_links.append(url)

    news_articles = []
    number_of_news = 3
    for link in news_links[:number_of_news]:  # Only consider the top 3 news articles
        article_res = requests.get(link, headers=headers).text
        article_soup = BeautifulSoup(article_res, "html.parser")

        article_content = ""
        for paragraph in article_soup.find_all("p"):
            article_content += paragraph.get_text() + "\n"

        news_articles.append((link, article_content))
    
    # Show the articles
    for i, (link, content) in enumerate(articles, start=1):
        print(f"Article {i}:")
        print(f"Link: {link}\n")
        print(content)
        print("=" * 80)
        print()

    return news_articles

######################################################################################################################

# Get the stock analysis of the company
def get_stock_analysis(query):
    Company_name,ticker=get_company_name_and_ticker(query)
    print({"Query":query,"Company_name":Company_name,"Ticker":ticker})
    stock_price_data=get_stock_price(ticker,history=10)
    stock_balance_sheet=get_company_balance_sheet(ticker)

    # stock_news=get_recent_stock_news(Company_name) -- not using news (too many tokens)
    # available_information=f"Stock Price: {stock_data}\n\nStock Financials: {stock_financials}\n\nStock News: {stock_news}"

    available_information=f"Stock Price: {stock_price_data}\n\nStock Financials: {stock_balance_sheet}"

    print("\n\nAnalyzing...\n")

    # With API
    analysis=llm(f"Please Provide a detail stock analysis, utilize the available data that are provided and generate investment advice. \
             User question: {query} \
             The company name is {Company_name} and the available information about this company is as follow: \
             {available_information}.\
             Please Provide a three pointwise investment analysis for this company (limit this to 2 sentences) to answer user question. At the end conclude with proper decision of how good of a stock it is to invest from a scale of 1 to 10(Start by addressing ur conclusion in this format Investment Rating x/10: xxxx). Try to Give positives and negatives.\
             Must not exceed 500 words\
             Imporatantly,Please do not provide any warning such as 'It is recommended to conduct further research and analysis or consult with a financial advisor before making an investment decision' in the answer as user is fully aware already!"
             )

    # Without API
    # analysis = f"Please Provide a detail stock analysis, utilize the available data that are provided and generate investment advice. \
    #          User question: {query} \
    #          The company name is {Company_name} and the available information about this company is as follow: \
    #          {available_information}.\
    #          Please Provide a three pointwise investment analysis for this company (limit this to 2 sentences) to answer user question. At the end conclude with proper decision of how good of a stock it is to invest from a scale of 1 to 10(Start by addressing ur conclusion in this format Investment Rating x/10: xxxx). Try to Give positives and negatives.\
    #          Must not exceed 500 words\
    #          Imporatantly,Please do not provide any warning such as 'It is recommended to conduct further research and analysis or consult with a financial advisor before making an investment decision' in the answer as user is fully aware already!"

    return analysis

if __name__ == "__main__":
    while True:
        query = input("What is your investment question?: ")
        
        if query == "exit":
            break
            
        analysis = get_stock_analysis(query)
        print(analysis)
        print("\n")
