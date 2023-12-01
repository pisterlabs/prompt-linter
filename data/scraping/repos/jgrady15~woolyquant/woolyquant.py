# Native Python Modules
import os
import json
import ast
from multiprocessing import Pool
from multiprocessing import cpu_count

# Dependencies
import pandas as pd
import openai 
import yfinance as yf
import numpy
import psycopg2
from pandas_datareader import data as pdr
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from dotenv import load_dotenv # We will deprecate this once we integrate AWS KMS

# Local Files
# from bindings import stockPair
from chatgpt import ChatGPT
from smartpair import SmartPair

def main():
    # NOTE: Eventually we want to pull in stock information from the users portfolio from Robinhood
    # conn = psycopg2.connect("dbname=")
    ticks = ["DPZ", "AAPL", "GOOGL", "GOOG", "BABA", "JNJ", "JPM", "BAC", "TMO", 
             "AVGO", "CVX", "DHR", "V", "MA", "COST", "CRM", "DIS", "CSCO", "QCOM", "AMD", 
             "GME", "SPY", "NFLX", "BA", "WMT", "GS", "XOM", "NKE", "META", "BRK-A", 
             "BRK-B", "MSFT", "AMZN", "NVDA", "TSLA"]
    
    sp = SmartPair()
    gpt = ChatGPT()

    # Initialize program
    sp.get_stock_data()
    sp._read_csv(ticks)
    
    # Covariance = Delta(Return ABC - Average ABC) * (Return XYZ - Average XYZ) / (Sample Size) - 1
    # print("-------- COVARIANCE/CORRELATION/ADF TEST/P-VAL BETWEEN TWO STOCK PAIRS --------")

    # sp.align_time_series()

    # Step 1: Align all possible time series.
    # Step 2: Calculate Covariance
    # Step 3: Calculate Correlation between stock pairs
    # Step 4: Determine if correlation is high
    # Repeat Step 1 otherwise go to step 5
    
    # Step 5: Calculate OLS Regression Model for highly correlated stock pairs
    sample_size = len(sp.price_return[ticks[0]])
    for i in range(0, len(ticks) - 1):
        for j in range(i + 1, len(ticks)):
            left, right = ticks[i], ticks[j]
            pair = f'{left}-{right}'

            covariance = sp.generate_covariance(sample_size=sample_size, left_tick=left, right_tick=right)
            
            corr = covariance / (sp.std_dev[left] * sp.std_dev[right])

            # Check if the correlation is really high before we decide pairs
            if corr > 0.79:
                # Cointegration
                model = OLS(sp.percent_return[left], sp.percent_return[right])
                results = model.fit()

                residuals = results.resid
                regression_slope = results.params
                adf_test = adfuller(residuals)
                p_val = adf_test[1]
                
                # sp.spread_series[pair] = stockPair.calculateAverageSpread(sample_size, sp.price_return[left], sp.price_return[right])

                parameters = f'{pair}: {covariance} {corr} {adf_test[0]} {p_val} {regression_slope} {sp.spread_series[pair]} {sp.mean_price_return[left]} {sp.mean_price_return[right]}'
                print(parameters)
                # gpt.chatGPT_conversation(pair=pair, parameters=parameters)
                
                # # Price
                # numpy.savetxt(f'./res/data/stocks/{left}-price-series.csv', sp.price_return[left], delimiter=",")
                # numpy.savetxt(f'./res/data/stocks/{right}-price-series.csv', sp.price_return[right], delimiter=",")
                
                # # Percent
                # numpy.savetxt(f'./res/data/stocks/{left}-percent-series.csv', sp.percent_return[left], delimiter=",")
                # numpy.savetxt(f'./res/data/stocks/{right}-percent-series.csv', sp.percent_return[right], delimiter=",")        

if __name__ == "__main__":
    yf.pdr_override()
    main()