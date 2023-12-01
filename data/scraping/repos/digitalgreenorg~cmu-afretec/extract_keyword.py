# This script extracts keywords from dataset by getting values from columns with unique values < 5
# and by extracting keywords from generated summary/ description

# Imports
import os
import re
from dotenv import load_dotenv
from typing import List
import pandas as pd
from langchain.llms import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_tags(description: str, dataset_path: str) -> List:

    llm = OpenAI(temperature=0)

    if dataset_path.endswith('.csv'):
        data_file = pd.read_csv(dataset_path)
    
    elif dataset_path.endswith('.xlsx'):
        data_file = pd.read_excel(dataset_path)
    
    string_data = data_file.select_dtypes(include=["object"])
    data_unique_num = string_data.nunique()
    select_data = data_unique_num[data_unique_num < 7]

    tags = []
    unique_values = []

    select_columns = select_data.keys()

    for column in select_columns:
        unique_values.extend(data_file[column].unique())

    unique_values = [ value for value in unique_values if isinstance(value, str) and len(value) < 16]

    if  len(unique_values) != 0:
        f_query = " Pick relevant Agricultural keywords from this string "
        context = f"{' '.join(unique_values)}"
        tags = llm(context + f_query).replace('\n', '').split(', ')

    query = "Provide the top 10 most relevant keywords related to agriculture from this description"

    description_tags = llm(description + query)

    # extract string to a list of keywords
    pattern = r'\n|\d+. '
    results = re.split(pattern, description_tags)
    description_tags_list = list(filter(None, results))

    tags.extend(description_tags_list)

    return tags

if __name__ == '__main__':
    extract_tags()