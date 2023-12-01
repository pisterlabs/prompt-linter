# Import necessary libraries
import os
import openai
import pandas as pd
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain import PromptTemplate


# Load environment variables from a .env file
load_dotenv(".env")

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Define the path to the CSV training data
train_data = "example_data.csv"


# Define a function to predict stock prices
def predict_stock(open_price, high_price, low_price, volume):

    # Initialize the OpenAI language model (GPT-3)
    llm = OpenAI(api_key=openai_api_key, temperature=0.4)
    
    # Create an agent that uses the language model and CSV data for predictions
    agent = create_csv_agent(llm, train_data, verbose=True)

    # Define the input string for the prediction task
    input_string = (
        "Read the CSV file and predict the future price of a stock based on "
        "Open Price: {}, High Price: {}, Low Price: {}, and Volume: {} if the values are "
        "in the dataset, if not then predict a value based on the {}"
    )

    # Format the input string with the provided stock data
    input_string = input_string.format(open_price, high_price, low_price, volume, train_data)

    # Run the agent to generate a prediction
    prediction = agent.run(input_string)
    
    return prediction