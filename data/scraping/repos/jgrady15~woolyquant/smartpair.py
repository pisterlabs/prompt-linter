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
from pandas_datareader import data as pdr
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from dotenv import load_dotenv # We will deprecate this once we integrate AWS KMS

# import stock_pair   

class SmartPair:
    def __init__(self) -> None:
        self.equity_info = {}
        self.percent_return = {}
        self.price_return = {}
        self.mean_percent_return = {}
        self.mean_price_return = {}
        self.covariance = {}
        self.std_dev = {}
        self.spread_series = {}

    def _read_csv(self, ticks) -> None:
        for t in ticks:
            pdr.get_data_yahoo(t, start="2019-01-01", end="2023-08-04").to_csv('./res/data/stocks/' + t + '.csv')
    
    def get_stock_data(self) -> None:
        path = "./res/data/stocks"
        for file in os.listdir(path):
            full_path = f'{path}/{file}'
            df = pd.read_csv(full_path)

            # For loop runs until EOF for ea. file. --- Big-O: O(n)
            # We convert from Dataframe to series for ea. file
            instrument = f'{file}'.split('.csv')[0]

            self.equity_info[instrument] = df

            # Pulls in the adj close price in percentage form
            percent_data = df['Adj Close'].pct_change().dropna() * 100

            # Pulls in the adj close price in normal value form
            value_data = df['Adj Close'].dropna()
            
            # Calculates standard deviation of Adj Close values, removes any NaN values, and rounds to 4 decimal places
            self.std_dev[instrument] = percent_data.std()

            # We then store the value in list form and round to pretty our data
            self.percent_return[instrument] = percent_data.to_numpy()
            self.price_return[instrument] = value_data.to_numpy()

            # Then we calculate the mean percentage as part of the covariance
            self.mean_percent_return[instrument] = percent_data.mean()
            self.mean_price_return[instrument] = value_data.mean()
    
    # NOTE: Returns an aligned time series list of stock pairs, where [0] is the left_equity and [1] is the right_equity 
    def align_time_series(self, left_equity, right_equity) -> list:
        curr_start = left_equity['Date'].iloc[0]
        curr_end = left_equity['Date'].iloc[-1]

        next_start = right_equity['Date'].iloc[0]
        next_end = right_equity['Date'].iloc[-1]

        start = max(curr_start, next_start)
        end = min(curr_end, next_end)

        return list()

    def generate_covariance(self, sample_size: int, left_tick: str, right_tick: str) -> float:
        pass
        # return stock_pair.calculateCovariance(sample_size, 
        #                                       self.percent_return[left_tick], 
        #                                       self.percent_return[right_tick], 
        #                                       self.mean_percent_return[left_tick], 
        #                                       self.mean_percent_return[right_tick])