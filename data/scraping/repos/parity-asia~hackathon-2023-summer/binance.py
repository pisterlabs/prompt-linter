
import os
import sys
import logging
import pandas as pd
import openai as openai

from fastapi import HTTPException
import datetime
import secrets
import json

from core.config import Config
from services.binance import binance_api
from services.helpers.chat_tools import data_summary
from services.helpers.index_engine import establish_private_dataset_index
from services.helpers.query_engine import private_dataset_query

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

import plotly.graph_objects as go

current_directory = os.path.dirname(os.path.realpath(__file__))
backend_directory = os.path.abspath(os.path.join(current_directory,"..",".."))
sys.path.insert(0, backend_directory)

os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
MODEL_NAME = Config.MODEL_NAME
LLM = OpenAI(model_name=MODEL_NAME, temperature=0)  

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


BINANCE_PROMPT = Config.BINANCE_PROMPT

def get_binance_prams(x):

    response_schemas = [
        ResponseSchema(name="symbol", description="'symbol' represents the query parameters extracted from the user input. For example, in the sentence 'What news are there about Bitcoin?', the query parameter would be 'BTC'."),
        ResponseSchema(name="k_lines", description="'k_lines' stands for K-line intervals, with the default being '1h'. Here are some examples, 'Last week': '1d'.'Last month': '1d'.'Yesterday': '1h'.'In 3 hours': '5m'.'Last hours': '5m'.'Two years ago': '3d'.If there is no time information, please provide '1h' as the answer. Please only provide the answer of the interval in the required format, don't explain anything."),
        ResponseSchema(name="dataframe", description="'dataframe' stands Extract time info from the provided text and translate to a time interval [start, end], expressed using python module `datetime.datetime.now()` and `datetime.timedelta()`. Here are some examples, 'Last week': '[datetime.datetime.now() - datetime.timedelta(weeks=1), datetime.datetime.now()]'.'Last month': '[datetime.datetime.now() - datetime.timedelta(days=30), datetime.datetime.now()]'.'Yesterday': '[datetime.datetime.now() - datetime.timedelta(days=1), datetime.datetime.now()]'.'In 3 hours': '[datetime.datetime.now(), datetime.datetime.now() + datetime.timedelta(hours=3)]'.'Two years ago': '[datetime.datetime.now() - datetime.timedelta(days=730), datetime.datetime.now()]'.If there is no time information, please provide '[]' as the answer. Please only provide the answer of the interval in the required format, don't explain anything."),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=BINANCE_PROMPT+"\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    model = LLM

    _input = prompt.format_prompt(question=x)
    output = model(_input.to_string())
    result = output_parser.parse(output)

    print("INFO:     BINANCE PRAMS:", result)

    return result

def cex_agent(question: str) -> str:

    try:
        params = get_binance_prams(question)
    except Exception as e:
        logger.error("ERROR:    Getting Binance parameters failed: %s", str(e))
        raise HTTPException(status_code=200, detail=str(e))

    symbol = params["symbol"]
    currency = "USDT"
    # klines = params["k_lines"]
    klines = "30m"
    dataframe = params["dataframe"]


    # dataframe = [datetime.datetime.now() - datetime.timedelta(days=30), datetime.datetime.now()]

    # print("symbol:", symbol)
    # print("currency:", currency)
    # print("klines:", klines)
    # print("dataframe:", dataframe)
    
    try:
        data = binance_api.get_historical_price(symbol, currency, klines, dataframe)
    except Exception as e:
        logger.error("ERROR:    Getting historical price from Binance failed: %s", str(e))
        raise HTTPException(status_code=200, detail=str(e))


    # print("GET BINANCE DATA:",data)
    
    df = pd.DataFrame(data)
   
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['quote_asset_volume'] = df['quote_asset_volume'].astype(float)
    df['taker_buy_base_asset_volume'] = df['taker_buy_base_asset_volume'].astype(float)
    df['taker_buy_quote_asset_volume'] = df['taker_buy_quote_asset_volume'].astype(float)

    
    df['num_trades'] = df['num_trades'].astype(int)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    
    image_fullpath = painting(df)

     # Downsample the data by half for the return
    df_half = df.iloc[::3]  

    ans_dict = {
        "question": question,
        "data": df_half.to_dict(orient='records'),
        "image_link": image_fullpath
    }

    # print("---------binance ans dic",ans_dict)

    # Convert the dictionary to a JSON string
    ans_string = json.dumps(ans_dict)

    result = data_summary(ans_string)

    return result

# Function to generate a random string
def generate_random_string(length):
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    random_string = ''.join(secrets.choice(alphabet) for _ in range(length))
    return random_string


def painting(df):

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='green',  # Green color for the rising candle
        decreasing_line_color='red'     # Red color for the falling candle
    )])

    fig.update_layout(
        title={
            'text': 'Candlestick Chart - Price Trend',  # Set the title of the graph
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis_title='Price',  # Set the title of y-axis
        xaxis_title='Date',  # Set the title of x-axis
        xaxis_rangeslider_visible=False,  # Hide the range slider
        autosize=False,
        width=1600,  # Set the width of the graph to 1600
        height=900,  # Set the height of the graph to 900
        plot_bgcolor='rgb(34, 34, 34)',  # Set the graph background color to dark
        paper_bgcolor='rgb(34, 34, 34)',  # Set the paper background color to dark
        margin=go.layout.Margin(
            l=50,  # left margin
            r=50,  # right margin
            b=100,  # bottom margin
            t=100,  # top margin
            pad=10  # padding
        )
    )

    fig.update_layout(
        title_font_family="Courier",
        title_font_color="white",  # Set the title color to white for readability on a dark background
        title_font_size=30,
        yaxis=dict(
            tickfont=dict(
                family='Courier',
                color='white',  # Set y-axis label color to white for readability on a dark background
                size=14
            ),
        ),
        xaxis=dict(
            tickfont=dict(
                family='Courier',
                color='white',  # Set x-axis label color to white for readability on a dark background
                size=14
            ),
        )
    )

    # Update x-axis and y-axis grid color and grid width
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgb(68, 68, 68)')  # Change x-axis grid line color and adjust grid line width
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgb(68, 68, 68)')  # Change y-axis grid line color and adjust grid line width

    # Generate current time accurate to the hour
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H")

    # Generate a random string of length 8
    random_string = generate_random_string(8)

    # Include the current time and random string in the image name for differentiation
    image_filename = f"chart_{current_time}_{random_string}.png"
    image_fullpath = f"static/image/{image_filename}"

    # Save the graph as a png file
    fig.write_image(image_fullpath)

    return image_fullpath



