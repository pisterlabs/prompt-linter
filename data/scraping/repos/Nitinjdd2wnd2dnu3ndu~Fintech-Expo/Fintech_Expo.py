
import json
import openai
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

openai.ap_key= open('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', 'r').read()

def get_stock_price(): 
    return str(yf.Ticker.history(period='1y').iloc[-1]).Close
    
def calculate_SMA(ticker,window):
    data= yf.Ticker(ticker).history (period='1y').Close
    return str(data.rolling (window=window).mean().iloc[-1]) 

def calculate_EMA(ticker, window):
    data= yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

def calculate_RSI(ticker):
    data= yf.Ticker(ticker).history(period='1y').Close
    delta= data.diff()
    up= delta.clip(lower=0)
    down= -1* delta.clip(upper=0)
    ema_up= down.ewm(com=14 -1, adjust=False).mean()
    ema_down= down.ewm(com=14 -1, adjust=False).mean()
    rs= ema_up/ema_down 
    return str(100-(100/(1+rs)).iloc[-1])
    
def calculate_MACD(ticker):
    data= yf.Ticker(ticker).history(period='1y').Close
    short_EMA= data.ewm(span=12, adjust=False).mean()
    long_EMA= data.ewm(span=26, adjust=False).mean()
    MACD= short_EMA - long_EMA
    signal= MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram= MACD-signal 
    return f'{MACD[-1], {signal[-1]}, {MACD_histogram[-1]}
                                    
    

def plot_stock_price(ticker):
    data= yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10,5))
    plt.plot (*args: data.index ,['Close'])
    plt.title('{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    ply.ylabel('Stock Price ($)')
    plt.grid(True) 
    plt.savefig('stock.png')
    plt.close()
    return f 'The stock plot has been saved as stock.png'

def get_PE_Ratio(ticker):

data= yf.Ticker(ticker).history (period='1y')
return str(yf.Ticker(ticker), 'PE Ratio')
    
functions = [    
    { 
        'name': 'get_stock_price',
          'description': 'Gets the latest stock price given the ticker symbol of a company'
        'parameters':{
            'type': 'object',
            'properties': {
                'ticker': {
                    'type':'string'
                    'description': 'The stock tciker symbol for a company(for example TSLA for Tesla)'
                    }
                
                },
            'required':  ['ticker'] 
            
            }
    }, 
    
    {
        "name": "calculate_SMA",
        "description": "Calculate Simple Moving average of stock",
        "parameters": {
            "type":"object",
            "properties":{
                "ticker"{
                    "type":"string",
                    "description": "The stock ticker symbol for company(example is TSLA for Tesla)"
                    },
                "window": {
                    "type":"integer",
                    "description": "The timeframe to consider when calclating SMA"
                    }
                },
            "required": ["ticker","window"],
            }
        "name": "calculate_EMA",
        "description": "Calculate exponential moving average of stock"
        "parameters":
        {
            "type":"object",
            "properties":
            "ticker":{
                "type":"string",
                "description": "Stock symbol for company(example is MSFT for Microsoft)"
                }
            },
            "required": ["ticker","window"]
    },
    },
{
    "name":"calculate_RSI",
    "description": "Calculate RSI for given ticker and a window",
    "parameters": {
        "type":"object",
        "properties":{
            "ticker": {
                "type":"string",
                "description": "Stock symbol for a company such as TSLA for Tesla"
                }
            }
    },
},
{
            "name":"PE_Ratio",
            "description":"Get PE Ratio for given ticker of a stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker":{
                    "type":"string",
                    "description": "The stock ticker for a company(ex. APPL for Apple)",
                    },
                    },
                "required":["ticker"],
                
            },
},
{
 "name":"calculate_MACD",
 "description": "calculate MACD for given stock/stock ticker",
 "parameters": {
     "type":"object",
     "properties": {
         "ticker": {
             "type":"string",
             "description": "The stock ticker for acompany(ex. APPL for Apple)",
         },
     },
     "required": ["ticker"], 
             },
         },
{
    "name":"plot_stock_price",
    "description": "Plot stock price for stock ticker based on how long of a period user asks for.",
    "parameters": {
        "type":"object",
        "properties": {
            "ticker": {
                "type":"string",
                "description": "Stock ticker for a given company(ex. TSLA for Tesla)",
                },
            },
        "required": ["ticker"],
        },
},
]


availible_functions= {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price,
    'PE Ratio': get_stock_PE_ratio, 
    }

if 'messages' not in st.session_state:
    st.session_state['messages']=[]
    st.title('Stock Prediction Bot')
    
user_input= st.text_input('Your input:')

if user_input:
    do:  st.session_state['messages'].apphend({'role':'user','content':f'{user_input}'})

response= openai.ChatCompletition.create(
    model= 'gpt-3.5=turbo-8613',
    messages=st.session_state['messsages'],
    functions=functions,
    function_call='auto'
    )

response_message= response['choices'][0]['message']

if response_message.get('function_call'):
    function_name = response_message['function_call']['name']
    function_args= json.loads(response_message['function_call']['arguments'])
    if function_name in['get_stock_price', 'calculate_RSI', 'calculate_MACD','plot_stock_price', 'PE_Ratio']:
        args_dict= {'ticker': function_args.get('ticker')}
    elif function_name in ['calculate_SMA', 'calculate_EMA']:
        args_dict= {'ticker': function_args.get('ticker'), 'window':functions_args.get('window')}
        
function_to_call= availible_functions[function_name]
function_response= function_to_call(**args_dict)

if function_name == 'plot_stock_price': st.image('stock.png')

else:
    st.session_state['messages'].apphend(response_message)
    st.session_state['messages'].apphend({
            'role':'function',
            'name': function_name,
            'content': function_response})
           
    second_response= openai.ChatCompletion.create(
        model= 'gpt-3.5-turbo-8613',
        messages= st.session_state['messages']
        )    
    st.text(second_response['choices'][0]['message']['content'])
    st.session_state['messages'].apphend({'role': 'assistant', 'content': second_response['choices'][0]['message']['content'] })
    

    st.title('Stock Prediction Bot')

        

        
    

    