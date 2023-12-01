import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
import seaborn as sns

def create_price_volume_chart(ticker, name):
    data = yf.download(ticker,'2016-01-01',date.today().strftime("%Y-%m-%d"))

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=data.index, y=data["Adj Close"], name="Price", line=dict(color="#BB86FC")),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(x=data.index, y=data["Volume"], name="Volume", opacity=0.5,
               marker=dict(line_width=0, line_color="#03DAC5", color="#03DAC5")),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text=f"{name} Price Chart",
        template="plotly_dark",
        autosize=False,
        width=700,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(yanchor="bottom", y=-0.3, xanchor="left", x=0.4, orientation="h"),
        bargap=0,
        bargroupgap=0,
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Price</b>", secondary_y=False, range=[0.5 * min(data["Adj Close"]), 1.1 * max(data["Adj Close"])])
    fig.update_yaxes(title_text="<b>Volume</b>", secondary_y=True, range=[0, 3.5 * max(data["Volume"])])

    # fig.show()
    # iplot(fig)
    # fig.write_image("output_chart.png", scale=2)

    return json.dumps({
        "instructions": "A plot has been generated and shown to the user, start your response by saying the requested chart can be seen above.",
        "action_taken": f"This is a message for you, not the user. You are integrated in an interactive finance tool. This tool has just shown the requested chart to the user. Please tell the user he can see the chart above and ask him what he wants to know next."
    }), fig

def create_price_chart(ticker, name):
    data = yf.download(ticker,'2016-01-01',date.today().strftime("%Y-%m-%d"))

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=data.index, y=data["Adj Close"], name="Price", line=dict(color="#BB86FC")),
        secondary_y=False,
    )

    # Add figure title
    fig.update_layout(
        title_text=f"{name} Price Chart",
        template="plotly_dark",
        autosize=False,
        width=700,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(yanchor="bottom", y=-0.3, xanchor="left", x=0.4, orientation="h"),
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Price</b>", secondary_y=False, range=[0.5 * min(data["Adj Close"]), 1.1 * max(data["Adj Close"])])

    # fig.show()
    # iplot(fig)
    # fig.write_image("output_chart.png", scale=2)

    return json.dumps({
        "instructions": "A plot has been generated and shown to the user, start your response by saying the requested chart can be seen above.",
        "action_taken": f"This is a message for you, not the user. You are integrated in an interactive finance tool. This tool has just shown the requested chart to the user. Please tell the user he can see the chart above and ask him what he wants to know next."
    }), fig

def create_volume_chart(ticker, name):
    data = yf.download(ticker,'2016-01-01',date.today().strftime("%Y-%m-%d"))

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Add traces
    fig.add_trace(
        go.Bar(x=data.index, y=data["Volume"], name="Volume", opacity=0.5,
               marker=dict(line_width=0, line_color="#03DAC5", color="#03DAC5")),
        secondary_y=False,
    )

    # Add figure title
    fig.update_layout(
        title_text=f"{name} Price Chart",
        template="plotly_dark",
        autosize=False,
        width=700,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(yanchor="bottom", y=-0.3, xanchor="left", x=0.4, orientation="h"),
        bargap=0,
        bargroupgap=0,
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Volume</b>", secondary_y=False)

    # fig.show()
    # iplot(fig)
    # fig.write_image("output_chart.png", scale=2)

    return json.dumps({
        "instructions": "A plot has been generated and shown to the user, start your response by saying the requested chart can be seen above.",
        "action_taken": f"This is a message for you, not the user. You are integrated in an interactive finance tool. This tool has just shown the requested chart to the user. Please tell the user he can see the chart above and ask him what he wants to know next."
    }), fig


def context_fetcher(ticker):
    directory = f'{ticker}_articles'.lower()
    articles = os.listdir(directory)
    #normally, can save articles intelligently so we can iteratively read title, date_time, content, source and save this overhead
    article_context = {
      "description": f"This JSON file contains a list of recent news articles about {ticker} company",
      "articles": [   
        {
          "title": "Ford, GM see strong US consumer demand for vehicles",
          "date_time": "June 15, 2023 10:52 AM",
          "content": open(f'{directory}/{articles[0]}','r',encoding='utf-8-sig').read().replace('\n',''),
          "source": "https://www.reuters.com/business/autos-transportation/ford-cfo-sees-supply-disruptions-easing-retail-prices-softening-2023-06-15/"
        },
        {
          "title": "General Motors Company (GM) Shows Fast-paced Momentum But Is Still a Bargain Stock",
          "date_time": "Fri, June 16, 2023 at 5:50 AM PDT",
          "content": open(f'{directory}/{articles[1]}','r',encoding='utf-8-sig').read().replace('\n',''),
          "source": "https://finance.yahoo.com/news/general-motors-company-gm-shows-125007782.html"
        },
        {
          "title": "Analysis: GM could reap billions by building combustion trucks and SUVs through 2035",
          "date_time": "June 13, 2023 5:26 AM PDT",
          "content": open(f'{directory}/{articles[2]}','r',encoding='utf-8-sig').read().replace('\n',''),
          "source": "https://www.reuters.com/business/autos-transportation/gm-could-reap-billions-by-building-combustion-trucks-suvs-through-2035-2023-06-13/"
        },
      ]
    }
    return f"Please provide a relevant answer to the user's question using the article contents in the JSON below, referencing most relevant sources (for example Jacobson or Reuters, ) :\n{json.dumps(article_context)}"

@st.cache_resource
def generate_index(pdf_folder_path, xy):

    PA = 'index.pkl'
    
    loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in xy]
    index = VectorstoreIndexCreator().from_loaders(loaders)
    return index


def financial_reports_answerer(ticker, prompt):
    pdf_folder_path = 'GM_AnnualReport'
    xy = os.listdir(pdf_folder_path)
    xy = [a for a in xy if 'pdf' in a]
    xy = [a for a in xy if int(a.split('_')[2].split('.')[0]) > 2015]

    print(xy)
    # print(loaders)
    index = generate_index(pdf_folder_path, xy)
    output = index.query(prompt)
    return output, f"User's question was answered based on the latest Financial Report. Keep this response in context for rest of the conversation :\n{output}"

## Rishi
import requests
import json

def get_jsonparsed_data(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for any non-successful status codes

    parsed_data = json.loads(response.text)
    return parsed_data


def get_financial_statements(ticker, limit, period, statement_type):
    if statement_type == "Income Statement":
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period={period}&limit={limit}&apikey=REPLACE"
    elif statement_type == "Balance Sheet":
        url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey=REPLACE"
    elif statement_type == "Cash Flow":
        url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey=REPLACE"
    
    data = get_jsonparsed_data(url)

    if isinstance(data, list) and data:
        return pd.DataFrame(data)
    else:
        st.error("Unable to fetch financial statements. Please ensure the ticker is correct and try again.")
        return pd.DataFrame()
    

def create_revenue_comparator(ticker1, ticker2):
    data1 = get_financial_statements(ticker1, 15, 'Annual', 'Income Statement')
    data2 = get_financial_statements(ticker2, 15, 'Annual', 'Income Statement')

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=data1['calendarYear'], y=data1['revenue'], name="Price"),
        secondary_y=False,
    )
    
        # Add traces
    fig.add_trace(
        go.Scatter(x=data2['calendarYear'], y=data2['revenue'], name="Price"),
        secondary_y=False,
    )

    # Add figure title
    fig.update_layout(
        title_text=f"{ticker1}-{ticker2} Price Chart",
        template="plotly_dark",
        autosize=False,
        width=700,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(yanchor="bottom", y=-0.3, xanchor="left", x=0.4, orientation="h"),
        bargap=0,
        bargroupgap=0,
    )

    return json.dumps({
        "instructions": "A plot has been generated and shown to the user, start your response by saying the requested chart can be seen above.",
        "action_taken": f"This is a message for you, not the user. You are integrated in an interactive finance tool. This tool has just shown the requested chart to the user. Please tell the user he can see the chart above and ask him what he wants to know next."
    }), fig


def create_growthchart(ticker, name):
    data1 = get_financial_statements(ticker, 15, 'Annual', 'Income Statement')

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Add traces
    print(data1)
    fig.add_trace(
        go.Scatter(x=data1['calendarYear'], y=data1[name], name="Price"),
        secondary_y=False,
    )

    # Add figure title
    fig.update_layout(
        title_text=f"Growth of {name} Chart",
        template="plotly_dark",
        autosize=False,
        width=700,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(yanchor="bottom", y=-0.3, xanchor="left", x=0.4, orientation="h"),
        bargap=0,
        bargroupgap=0,
    )

    return json.dumps({
        "instructions": "A plot has been generated and shown to the user, start your response by saying the requested chart can be seen above.",
        "action_taken": f"This is a message for you, not the user. You are integrated in an interactive finance tool. This tool has just shown the requested chart to the user. Please tell the user he can see the chart above and ask him what he wants to know next."
    }), fig


from scipy.cluster.hierarchy import single,complete,average,ward,dendrogram

def hierarchical_clustering(distance_matrix, sector, tickers, ax, method='complete'):
    
    labels = [tickers[col] for col in distance_matrix.columns]
    print(distance_matrix)
    if method == 'complete':
        Z = complete(distance_matrix)
    if method == 'single':
        Z = single(distance_matrix)
    if method == 'ward':
        Z = ward(distance_matrix)

    dn = dendrogram(Z,labels=labels,distance_sort=True, ax=ax)
    return

def universe_correlator(tickers, sector, hierarchical=False):
    
    price_data=yf.download(tickers=[*tickers.keys()],start='2016-01-01')['Adj Close']
    returns_data = price_data.pct_change(1).dropna(how='all',axis=0).replace([np.nan], 0)
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,6))
    
    if hierarchical:
        correlations = returns_data.corr().dropna(axis=0, how='all').dropna(axis=1, how='all')
        hierarchical_clustering(correlations,sector,tickers,ax,
                               method='single')
        ax.set_title(f"{sector} Hierarchical Clusters using Correlation Distance",fontsize=20)
    else:    
        sns.heatmap(returns_data.corr().rename(index=tickers,columns=tickers),vmin=0,vmax=1,cmap='OrRd',ax=ax)
        ax.set_title(f'{sector} Returns Correlations {returns_data.index[0].date()} onwards')
        
    return json.dumps({
        "instructions": "A plot has been generated and shown to the user, start the sector has been analysed.",
        "action_taken": f"This is a message for you, not the user. You are integrated in an interactive finance tool. This tool has just shown the requested chart to the user. Please tell the user he can see the chart above and ask him what he wants to know next."
    }), fig


FUNCTIONS = [
        {
            "name": "create_price_volume_chart",
            "description": "Create a Price & Volume chart for a specific stock ticker over the past few years",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The Yahoo Finance ticker of a stock, for instance GM for General Motors or T for T-Mobile. Also include the country suffix of a ticker if necessary, such as DBK.DE for Deutsche Bank and CDR.WA for CD Projekt Red",
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of a publicly listed company, either given directly or through the ticker, for instance General Motors for GM or T-Mobile for T"
                    }
                },
                "required": ["ticker", "name"],
            },
        },
        {
            "name": "create_price_chart",
            "description": "Create a Price chart for a specific stock ticker over the past few years",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The Yahoo Finance ticker of a stock, for instance GM for General Motors or T for T-Mobile. Also include the country suffix of a ticker if necessary, such as DBK.DE for Deutsche Bank and CDR.WA for CD Projekt Red",
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of a publicly listed company, either given directly or through the ticker, for instance General Motors for GM or T-Mobile for T"
                    }
                },
                "required": ["ticker", "name"],
            },
        },
        {
            "name": "create_volume_chart",
            "description": "Create a Volume chart for a specific stock ticker over the past few years",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The Yahoo Finance ticker of a stock, for instance GM for General Motors or T for T-Mobile. Also include the country suffix of a ticker if necessary, such as DBK.DE for Deutsche Bank and CDR.WA for CD Projekt Red",
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of a publicly listed company, either given directly or through the ticker, for instance General Motors for GM or T-Mobile for T"
                    }
                },
                "required": ["ticker", "name"],
            },
        },
        {
            "name": "context_fetcher",
            "description": "Whenever user asks for news on a company, read the recent articles around company's news. Cache them into context in past message history, use this context to inform answers to user questions regarding recent company news & sentiment. Function should only be called once! Check past message history, if article context already there use it to answer question and don't rerun this function!",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The Yahoo Finance ticker of a stock, for instance GM for General Motors or T for T-Mobile. Also include the country suffix of a ticker if necessary, such as DBK.DE for Deutsche Bank and CDR.WA for CD Projekt Red",
                    }
                },
                "required": ["ticker"],
            },
        },
        {
            "name": "financial_reports_answerer",
            "description": "Whenever user asks for answers on the earnings and financial reports on a company, call this function for factual answers on the same.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The Yahoo Finance ticker of a stock, for instance GM for General Motors or T for T-Mobile. Also include the country suffix of a ticker if necessary, such as DBK.DE for Deutsche Bank and CDR.WA for CD Projekt Red",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt user just asked.",
                    }
                },
                "required": ["ticker", "prompt"],
            },
        },
        {
            "name": "create_revenue_comparator",
            "description": "Create a Revenue comparator chart to compare a specific stock ticker to another",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker1": {
                        "type": "string",
                        "description": "The Yahoo Finance ticker of a stock, for instance GM for General Motors or T for T-Mobile. Also include the country suffix of a ticker if necessary, such as DBK.DE for Deutsche Bank and CDR.WA for CD Projekt Red",
                    },
                     "ticker2": {
                        "type": "string",
                        "description": "The Yahoo Finance ticker of a stock, for instance AAPL for Apple or T for T-Mobile. Also include the country suffix of a ticker if necessary, such as DBK.DE for Deutsche Bank and CDR.WA for CD Projekt Red",
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of a publicly listed company, either given directly or through the ticker, for instance General Motors for GM or T-Mobile for T"
                    }
                },
                "required": ["ticker", "name"],
            },
        },
        {
            "name": "create_growthchart",
            "description": "Create a chart for one of 'revenue, ebidta, grossProfit, netIncome, operatingExpenses' for a specific stock ticker over the past few years",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The Yahoo Finance ticker of a stock, for instance GM for General Motors or T for T-Mobile. Also include the country suffix of a ticker if necessary, such as DBK.DE for Deutsche Bank and CDR.WA for CD Projekt Red",
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the property - revenue, ebidta, grossProfit, netIncome, operatingExpenses"
                    }
                },
                "required": ["ticker", "name"],
            },
        },
        {
            "name": "universe_correlator",
            "description": """ Whenever user asks for competitors or comparisons within a sector, think of competitors you know, particularly each one's name and yahoo finance ticker. 
                               Then you will visualise a correlation plot or a hierarchical clustering plot to inform user. Only call the prompt asks for a comparison explicitly.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "string",
                        "description": """ Using your knowledge about the sector, generate a string mimicking python dictionary format '{"yahooticker1":"company1", "yahooticker2":"company2"}' and so on. Ensure yahoo tickers exist!! Choose at least 8 competitors, double check you have seen yahoo finance data for the ticker!! """,
                    },
                    "sector": {
                        "type":"string",
                        "description": "In a single word, summarise the defining common feature of these companies; e.g. Automotive or Healthcare",
                    },
                    "hierarchical":{
                        "type": "boolean",
                        "description": "If you think user wants return correlations set hierarchical=False, if you think user cares about structures and clusters use hierarchical=True"
                    }   
                },
                
                "required": ["tickers","sector","hierarchical"],
            },
        },

    ]
