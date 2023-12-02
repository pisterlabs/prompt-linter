from openai import InvalidRequestError
import streamlit as st
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os
import sys
from io import StringIO
import re
from IPython.display import Markdown, display
from openai.error import AuthenticationError,RateLimitError
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# all stats model

from statsmodels.tsa.api import  SARIMAX, VAR, VECM
# from statsmodels.tsa.statespace import (StructTS, DynamicFactor, 
#                                         MarkovRegression, LocalLevel, LocalLinearTrend)
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.api import OLS
from sklearn.linear_model import LinearRegression

import pandas as pd
from langchain.callbacks import get_openai_callback
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
# from statsmodels.m
from streamlit import components

def display_markdown(text):
    display(Markdown(text))



# Create a function to display terminal output in the UI

def display_output(output,df):
    cleaned_output = re.sub(r'\x1b\[[0-9;]*m', '', output)
    lines = cleaned_output.strip().split("\n")
    for line in lines:
        if line.startswith("Final Answer:"):
            value = line.split(":")[1].strip()
            st.text(f"Final Answer: {value}")
        elif line.startswith("fig.show()"):
            # Get the plot filename
            # Save the plot as an image file
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            plot_filename = f'plot/btc_plot_{timestamp}.html'

            # Save the plot with the generated filename
            plt.savefig(plot_filename)
            # # Display the saved plot using Streamlit
            # st.image(plot_filename)

            # st.pyplot("plt")
            # plot_path = 'save/plot.html'

        # Render the HTML plot in the Streamlit app
            st.components.v1.html(open( plot_filename, 'r').read(), width=800, height=600)

        else:
            st.text(line)

def list_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    return files

def generate_crypto_responses(crypto, start_date, end_date, prompt, key, chat_history, offline_data,selectedfile):
    # path=f"Online-Data/{crypto}.csv"
    if offline_data:
        # Load data from the offline file
        data = pd.read_csv(f"Offline-Data/{selectedfile}")
    else:
        # Download the cryptocurrency data
        data = yf.download(crypto, start=start_date, end=end_date)

        # Write data to a CSV file

        data.to_csv(f"Online-Data/{crypto}.csv")


        template="""

        You are working with a CSV file in Python that contains cryptocurrency data. The file name is {db}, and you should use the 'Date' column as the index when loading the data. Your task is to analyze the cryptocurrency data and answer questions related to it. You have access to various tools and libraries to assist you, including machine learning and time series modeling.


        python_repl_ast: A Python shell. Use this to execute python commands. Input should be a valid python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.

        To make your visualizations colorful and attractive, consider the following tips:
        - Choose vibrant and contrasting colors for different elements in your plots.
        - Experiment with different marker styles and line styles.
        - Customize the background, grid, and text styling to enhance visual appeal.
        - Pay attention to the layout and composition of your plots.

        
        To help you accomplish the task, here are some guidelines:

        - Familiarize yourself with the structure of the CSV file and the data it contains.
        - Take a hands-on approach to data processing and manipulation.
        - Ensure your code is free of syntax errors.
        - For machine learning tasks, use the provided dataset.
        - Use the following columns as features (X) for machine learning prediction: 'Open', 'High', 'Low', 'Close', 'Volume'. The target variable (Y) is 'Adj Close'. Exclude the 'Date' column from the training process.



        When visualizations are required, always utilize the 'plotly.graph_objects' library for plotting. Apply the tips mentioned earlier to create colorful and attractive plots. Use the `fig.show()` command to display the diagrams.

        fig.show()  # Display the diagram

        For machine learning tasks, the following libraries are available:
        1. LSTM and GRU from 'tensorflow.keras.layers'
        2. SARIMAX, VAR, and VECM from 'statsmodels.tsa.api'
        3. SimpleExpSmoothing and ExponentialSmoothing from 'statsmodels.tsa.holtwinters'
        4. AR from 'statsmodels.tsa.ar_model'
        5. seasonal_decompose from 'statsmodels.tsa.seasonal'


        Use the following format:

        Question:  the input question you must answer
        Thought:  you should always think about what to do
        Action:  the action to take, should be one of [python_repl_ast]
        Action Input:  the input to the action
        Observation:  the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times 
        
        Thought:  I now know the final answer
        Final Answer:  the final answer to the original input question

        This is the result of print(df.head()): {df_head}    

        Begin! Question: {input} {agent_scratchpad}"""

    # Create the Langchain agent
    llm = OpenAI(temperature=0, openai_api_key=key)
    agent = create_csv_agent(
        llm,
        f"Offline-Data/{selectedfile}" if offline_data else f"Online-Data/{crypto}.csv",verbose=True
    )
            # Prompt Template

    # agent.agent.llm_chain.prompt.input_variables.append('db')
    # print('input :', agent.agent.llm_chain.prompt.input_variables)
  




    if not offline_data:
        agent.agent.llm_chain.prompt.template=template


    # print("prompt",(agent.agent.llm_chain.prompt.template))

    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    # Run the agent to generate a response based on the user's prompt
    try :
        with get_openai_callback() as cb:
            if not offline_data:
                data = f"Online-Data/{crypto}.csv"
                print("file", data)
                agent.agent.llm_chain.prompt.input_variables.append('db')
                print('input :', agent.agent.llm_chain.prompt.input_variables)
                response = agent.run({'input': prompt,"db":data})
            else :
                response = agent.run({'input': prompt})

            # Restore stdout and display the captured output in the UI
            sys.stdout = old_stdout
            output = redirected_output.getvalue()
            with st.expander("**Detail AI Calculation**"):
                (display_output(output, data))
            chat_history.append((prompt, response))

            return response, output
    except InvalidRequestError:
        st.write("Max Token error. Give Prompting properly") 

def display_chat_history(chat_history):
    for i, (question, answer) in enumerate(chat_history):
        st.info(f"Question {i + 1}: {question}")
        st.success(f"Answer {i + 1}: {answer}")
        st.write("----------")

def download_data(crypto, start_date, end_date):
    data = yf.download(crypto, start=start_date, end=end_date)
    data.to_csv(f"Online-Data/{crypto}.csv")
    st.success(f"Data for {crypto} downloaded successfully!")

def main():
    st.title("Finlyzr: AI-Powered Financial Data Assistant")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []


    if 'show_dataframe' not in st.session_state:
        st.session_state.show_dataframe = False


    with st.sidebar.expander("API Key Input"):
        with st.form(key="api_form"):
            api_key = st.text_input("Enter your API key:", type="password")
            submit_button = st.form_submit_button(label="Submit")

            if submit_button and api_key:
                # Perform actions using the API key
                st.success("API key submitted:")

    if api_key:
        # Define the cryptocurrency options
        try:
            cryptos = [
                'BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'LTC-USD', 'DOGE-USD',
                'USDT-USD', 'ADA-USD', 'BNB-USD', 'LINK-USD', 'XLM-USD', 'SOL1-USD',
                'THETA-USD', 'ETC-USD', 'FIL-USD', 'TRX-USD', 'EOS-USD', 'XMR-USD',
                'AVAX-USD'
            ]
            with st.sidebar.expander("File Uploader"):
                datafiles = st.file_uploader("Upload CSV", type=['csv'], accept_multiple_files=True)
                
                if datafiles is not None:
                    for datafile in datafiles:
                        file_details = {"FileName": datafile.name, "FileType": datafile.type}
                        df = pd.read_csv(datafile)
                        file_path = os.path.join("Offline-Data", datafile.name)
                        df.to_csv(file_path)
                        st.success(f"Data for {datafile.name} downloaded successfully!")

            with st.sidebar.expander("Data Source"):
                data_source = st.radio("Select Data Source:", ("Online Data", "Offline Data"))
                offline_data = True if data_source == "Offline Data" else False
                # online_Data = True if data_source == "Online Data" else False


            if offline_data:
                with st.sidebar.expander("Fetch Offline Data"):
                    folder_paths = 'Offline-Data/'
                    files = list_files_in_folder(folder_paths)
                    selectedfile = st.selectbox("Select a file", files)
                    st.write("You selected this file:", selectedfile)
            else:
                selectedfile = None

            with st.sidebar.expander("Fetch Online Data"):
                # Get user input
                crypto = st.selectbox("Select a cryptocurrency", cryptos)
                start_date = st.date_input("Select a start date")
                end_date = st.date_input("Select an end date")

                # Add a "Download Data" button
                if st.button("Download Data"):
                    download_data(crypto, start_date, end_date)
                    st.session_state.show_dataframe = True
                    st.session_state.show_clear = True * 2

            st.session_state.show_clear = True

            if st.session_state.get("show_clear", False):
                if st.sidebar.button("Clear"):
                    st.session_state.show_dataframe = False
                    st.session_state.show_clear = False              

            if st.session_state.show_dataframe:
                if offline_data and selectedfile:
                    # Display offline data
                    df = pd.read_csv(f"Offline-Data/{selectedfile}")
                else:
                    # Display online data
                    df = pd.read_csv(f"Online-Data/{crypto}.csv")
                st.dataframe(df)

            with st.form(key='my_form', clear_on_submit=True):
                prompt = st.text_input("Query:", placeholder="Type your query", key='input')
                submit_button = st.form_submit_button(label='Send', type='primary')

            if submit_button and prompt:
                response,output = generate_crypto_responses(crypto, start_date, end_date, prompt, api_key,
                                                    st.session_state.chat_history, offline_data, selectedfile )

                # Display the question and response
                st.write(f"Question: {prompt}")
                st.write(f"Answer: {response}")

                # display_output(output)

            if st.sidebar.button("Show Data"):
                try :
                    if offline_data and selectedfile:
                        # Display offline data
                        df = pd.read_csv(f"Offline-Data/{selectedfile}")
                    else:
                        # Display online data
                        df = pd.read_csv(f"Online-Data/{crypto}.csv")
                    st.dataframe(df)
                except :
                    st.sidebar.warning(f"Please Download {crypto} data")

            with st.expander("**View Chat History**"):
                display_chat_history(st.session_state.chat_history)

            if st.button("Clear Chat History"):
                st.session_state['chat_history'] = []

        except AuthenticationError as e :
            link = "[Click here](https://platform.openai.com/account/api-keys)"
            st.error(f"Ensure the API key used is correct, clear your browser cache, or generate a new one {link}")

    else:
        st.warning("Please add an API key")


if __name__ == '__main__':
    main()
