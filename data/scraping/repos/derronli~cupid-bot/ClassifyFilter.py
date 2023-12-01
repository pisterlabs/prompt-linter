import pandas as pd
from CohereLayer import *

def classifyFilter(df):
    # for i in range
    # sleep 
    # cohere key to avoid rate limitations 

    # Apply the cohere classifier on each entry in the responses
    responses = df["Response"]
    response_predicts = responses.apply(classify_text).tolist()
    # Add the response_predicts to the dataframe
    df["Response Semantic"] = response_predicts

    output_file = "middleForTesting.csv"
    df.to_csv(output_file, index=False)

    # Filter the dataframe by removing all non-positive classifications
    # MAYBE TRY LEAVING IN NEUTRALS AS WELL 
    df = df[df["Response Semantic"] == 'positive']

    df = df.reset_index(drop=True)

    # Write the DataFrame to a CSV file
    output_file = "convClassFilter.csv"
    df.to_csv(output_file, index=False)

    return df






''' ==========================FUNCTION CALLS (for testing)=========================== '''
