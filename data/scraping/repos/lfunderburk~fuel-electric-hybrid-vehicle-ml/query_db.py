import requests
import zipfile
import io
import os
import pandas as pd
from sqlalchemy.engine import create_engine
import openai 
from dotenv import load_dotenv
from IPython.display import display, Markdown
from pathlib import Path

class Prompter:
    def __init__(self, api_key, gpt_model):
        if not api_key:
            raise Exception("Please provide the OpenAI API key")


        openai.api_key = os.environ.get("OPENAI_API_KEY")

        self.gpt_model = gpt_model

    def prompt_model_print(self, messages: list):
        response = openai.ChatCompletion.create(model=self.gpt_model, 
                                                messages=messages,
                                                temperature=0.2)
        display(Markdown(response["choices"][0]["message"]["content"]))
    
    def prompt_model_return(self, messages: list):
        response = openai.ChatCompletion.create(model=self.gpt_model, 
                                                messages=messages,
                                                temperature=0.2)
        return response["choices"][0]["message"]["content"]
    
def data_cleaner(df):
    df.columns = df.columns.str.replace('.', '_')

    # Replace the character '/' with '_per_' all entries
    df.columns = df.columns.str.replace('/', '_per_')

    # drop column hybrid_in_fuel	hybrid_in_electric	aggregate_levels	vehicle_type_cat
    df = df.drop(['hybrid_in_fuel', 'hybrid_in_electric', 'aggregate_levels','transmission_','fuel_type'], axis=1)

    return df

def init_database(df):
    # Set up engine
    engine = create_engine("sqlite://")
    df.to_sql("vehicleDB", engine)

    return engine

if __name__=="__main__":

    # Load the .env file
    load_dotenv(".env")

    # Get OpenAI API key
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    # Initialize prompter
    prompter = Prompter("gpt-3.5-turbo")

    # Get the current working directory
    current_working_directory = os.getcwd()

    # Convert the current working directory to a Path object
    script_dir = Path(current_working_directory)

    # Set the path to the raw data
    predicted_data_path = script_dir / 'data' / 'predicted-data' / 'vehicle_data_with_clusters.csv'

    # Set up engine
    engine = create_engine("sqlite://")
    df = pd.read_csv(predicted_data_path)

    # Data wrangling
    df.columns = df.columns.str.replace('.', '_')

    # Replace the character '/' with '_per_' all entries
    df.columns = df.columns.str.replace('/', '_per_')

    # drop column hybrid_in_fuel	hybrid_in_electric	aggregate_levels	vehicle_type_cat
    df = df.drop(['hybrid_in_fuel', 'hybrid_in_electric', 'aggregate_levels','transmission_','fuel_type'], axis=1)

    # Store the data in the database
    df.to_sql("vehicleDB", engine)

    data_query = "Show hybrid vehicles"
    sample_values = {df.columns[i]: df.values[0][i] for i in range(len(df.columns))}

    datagen_prompts_2 = [
        {"role" : "system", "content" : "You are a data analyst specializing in SQL, you are presented with a natural language query, and you form queries to answer questions about the data."},
        {"role" : "user", "content" : f"Please generate 1 SQL queries for data with columns {', '.join(df.columns)} and sample values {sample_values}. \
                                        The table is called 'vehicleDB'. Use the natural language query {data_query}"},
    ]


    prompter.prompt_model_print(datagen_prompts_2)

    result1 = prompter.prompt_model_return(datagen_prompts_2)

    with engine.connect() as connection:
        result = connection.execute(result1.split("\n\n")[0])
        for row in result:
            print(row)  


    