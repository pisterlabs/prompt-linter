# %%
import os 
import re
import openai
import backoff

import pandas as pd

from openai.error import RateLimitError
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain

def generate_table_summary(df):
    """
    Generate a summary table for a given DataFrame `df` containing counts 
    and average citations based on the 'gpt_label' and 'year_published' columns.
    
    Parameters:
    - df (pandas.DataFrame): The input DataFrame including the columns ["gpt_label", "year_published", "citations"].
    
    Returns:
    - pandas.DataFrame: A summarized DataFrame containing the 'gpt_label' column,
                        optional 'growth' column (if more than one year is present),
                        and 'avg_citations' column.
    """
    
    # Group by 'gpt_label' and 'year_published' columns and count the occurrences of each label in each year
    df_summary = df.groupby(["gpt_label", "year_published"]).agg({"gpt_label": "count"}).rename(columns={"gpt_label": "count"}).reset_index(drop=False)
    
    # Pivot the table to have 'gpt_label' as rows and 'year_published' as columns with count values
    df_summary = df_summary.pivot(index="gpt_label", columns="year_published", values="count").fillna(0).reset_index(drop=False)
    
    # Calculate growth as a percentage, comparing the last two years if more than one year of data is available
    if len(df.year_published.unique().tolist()) > 1:
        df_summary["growth"] = round(((df_summary.iloc[:, -1] / df_summary.iloc[:, -2]) * 100)-100,2)
    
    # Group by 'gpt_label' and calculate average citations for each label
    cite_summary = df.groupby("gpt_label").agg({"citations": "mean"}).reset_index(drop=False)
    
    # Merge the main summary table with the average citations data
    df_summary = df_summary.merge(cite_summary, on="gpt_label", how="left")
    
    # If more than one year of data is available, return 'gpt_label', 'growth', and 'avg_citations' columns.
    # Otherwise, just return 'gpt_label' and 'avg_citations' columns.
    if len(df.year_published.unique().tolist()) > 1:
        return df_summary[["gpt_label", "growth", "citations"]].rename(columns={"citations": "avg_citations"})
    else:
        return df_summary[["gpt_label", "citations"]].rename(columns={"citations": "avg_citations"})

def get_df_string(df):
    """
    Convert a DataFrame into a human-readable string representation suitable for GPT prompts.
    
    This function assumes one of two DataFrame structures:
    1) ["gpt_label", "growth", "citations"] when multi-year data is available.
    2) ["gpt_label", "citations"] when only single-year data is present.
    
    It converts the DataFrame into a string where each row is represented as:
    For multi-year data: 'Cluster: [gpt_label], Growth: [growth], Average citations: [citations]'
    For single-year data: 'Cluster: [gpt_label], Average citations: [citations]'
    
    Parameters:
    - df (pandas.DataFrame): The input DataFrame to be converted. Should contain 
                             columns ["gpt_label", "growth", "citations"] for multi-year data, or
                             columns ["gpt_label", "citations"] for single-year data.
    
    Returns:
    - str: A string representation of the DataFrame, formatted for GPT prompts.
    """
    
    df_string = ""  # Initialize an empty string to store the results

    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        
        # Start with the 'gpt_label' information
        text = f'Cluster: {row["gpt_label"]}, '
        
        # Check if "growth" column is present and add its value
        if "growth" in df.columns:
            text += f'Growth: {round(row["growth"], 2)}%, '

        # Append the average citations for the current row
        text += f'Average citations: {round(row["avg_citations"], 2)}'
        
        # Add the constructed row text to the final string with a newline
        df_string += text + "\n"

    return df_string

def get_topic_summary(df_string):
    """
    Generate a summarized description of a clustered dataset based on a string representation 
    using the OpenAI model. The summary focuses on topic importance based on publication growth 
    and average citations without detailing the cluster label names.
    
    Parameters:
    - df_string (str): A string representation of the DataFrame, typically output from get_df_string(), 
                       formatted with details on 'gpt_label', 'growth' (if multi-year), and 'citations'.
    
    Returns:
    - list: A list of sentences forming the summary.
    
    Notes:
    - The function leverages OpenAI's API (which must be accessible with a valid API key) 
      to generate a summary of the data.
    - The OpenAI model is primed with a specific prompt that describes the requirement for the 
      summary and what details to focus on.
    """
    
    # Initialize the OpenAI model with specific parameters
    llm = OpenAI(temperature=0.1, openai_api_key=os.getenv("OPENAI_TOPIC_CLUSTERING"))
    
    # Template used to instruct the OpenAI model about the nature of the data and what kind of summary is expected
    label_template = """
    The following is a summary of a clustered dataset providing the topic names, growth 
    if publications were provided for more than one year otherwise this will be absent 
    (indicating the change in publication output, positive figures mean the discipline was growing) 
    and average citations received per article published. Provide a summary of the table, 
    indicating which topics are the most important (defined by categories displaying the 
    most significant positive publication growth and high avg cites per article) and which 
    display lower value (those with low or negative growth and avg citations). Use numbers 
    and percentages in your summary. Please do not describe the names of the cluster labels.
        {text}
    """
    
    # Split the input string into manageable chunks for the OpenAI model
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(df_string)
    
    # Create a list of documents to pass to the OpenAI model
    docs = [Document(page_content=t) for t in texts]
    
    # Construct a prompt for the OpenAI model
    PROMPT = PromptTemplate(template=label_template, input_variables=["text"])
    
    # Load a predefined OpenAI processing chain
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    
    # Get the model's output summary
    output = chain.run(docs)
    
    # Split the output into individual sentences, while avoiding breaking on decimal points
    _list = re.split(r'(?<!\d)\.(?!\d|$)', output.replace("\n", ""))
    _list = [i.strip() for i in _list]
    
    return _list

def topic_summary(df):
    """
    Generate a summarized description of a DataFrame representing topic clusters.
    
    This function processes a given DataFrame to produce a human-friendly summary
    describing the importance of topics based on publication growth and average citations.
    The summary does not include specific topic names.
    
    Workflow:
    1. Summarize the DataFrame to generate statistics on growth and average citations.
    2. Convert the summarized DataFrame into a string representation.
    3. Pass the string representation to OpenAI's model to get a descriptive summary.
    4. Format the summary for output.
    
    Parameters:
    - df (pandas.DataFrame): An input DataFrame, expected to have certain structure and columns 
                             as consumed by the 'generate_table_summary' function.
    
    Returns:
    - list: A list of sentences forming the summary.
    """
    
    # Step 1: Summarize the input DataFrame
    table = generate_table_summary(df)
    
    # Step 2: Convert the summarized table into a string
    df_string = get_df_string(table)
    
    # Step 3: Obtain a human-friendly summary from OpenAI's model
    summary = get_topic_summary(df_string)
    
    # Step 4: Format the summary by ensuring each sentence ends with a period
    summary = [i + "." for i in summary if i[-1] != "."]
    
    return summary

def generate_comp_table_summary(df, comparator):
    """
    Generate a summarized table of topic clusters based on a specific comparator column value.
    
    This function aggregates and summarizes data for rows that match a specific 
    'full_source_title' comparator. It then computes the growth of publications for 
    each topic cluster and merges with average citation data.
    
    Parameters:
    - df (pandas.DataFrame): An input DataFrame containing data on publications.
                             Expected columns include ["gpt_label", "year_published", 
                             "full_source_title", "citations"].
    - comparator (str): The specific value in the 'full_source_title' column to filter 
                        the DataFrame on before summarizing.
    
    Returns:
    - pandas.DataFrame: A summarized DataFrame with columns ["gpt_label", "growth", "avg_citations"].
    
    Notes:
    - The function assumes that the 'year_published' column contains at least two unique years 
      to calculate growth.
    """
    
    # Filter the DataFrame based on the provided comparator for the 'full_source_title' column
    df_summary = df[df["full_source_title"] == comparator]
    
    # Group by 'gpt_label' and 'year_published', and count the occurrences of each 'gpt_label'
    df_summary = df_summary.groupby(["gpt_label", "year_published"]).agg({"gpt_label": "count"}).rename(columns={"gpt_label": "count"}).reset_index(drop=False)
    
    # Pivot the summarized data to have 'gpt_label' as rows and 'year_published' as columns
    df_summary = df_summary.pivot(index="gpt_label", columns="year_published", values="count").fillna(0).reset_index(drop=False)
    
    # Calculate the growth based on the counts from the last two years in the data
    df_summary["growth"] = round(((df_summary.iloc[:, -1] / df_summary.iloc[:, -2]) * 100) - 100, 2)
    
    # Compute the average citations for each 'gpt_label'
    cite_summary = df.groupby("gpt_label").agg({"citations": "mean"}).reset_index(drop=False)
    
    # Merge the growth data with average citation data
    df_summary = df_summary.merge(cite_summary, on="gpt_label", how="left")
    
    # Return a DataFrame with selected columns and renamed columns for clarity
    return df_summary[["gpt_label", "growth", "citations"]].rename(columns={"citations": "avg_citations"})


def comparator_analysis(topic_string, comparator_string):
    """
    Analyzes and compares a topic dataset with a comparator dataset using OpenAI's model.
    
    This function takes two string representations of datasets: one for a general topic and the other 
    for a specific comparator (like a specific journal title or region). It combines these strings, 
    sends them to an OpenAI model with a specified prompt to get a comparative analysis, and then
    processes the output to generate a list of sentences providing insights and recommendations.
    
    Parameters:
    - topic_string (str): String representation of the general topic dataset.
    - comparator_string (str): String representation of the comparator dataset.
    
    Returns:
    - list: A list of sentences forming the comparative analysis between the topic and comparator datasets.
    
    Notes:
    - The function assumes the input strings are already formatted in a way the OpenAI model understands.
    - It uses specific template prompts and configurations for the OpenAI model.
    """
    
    # Combine the topic and comparator strings with a separator
    search_string = topic_string + "//" + comparator_string
    
    # Initialize OpenAI's model with specific parameters
    llm = OpenAI(temperature=0.1, openai_api_key=os.getenv("OPENAI_TOPIC_CLUSTERING"))
    
    # Define the prompt template to instruct the model on how to process and compare the datasets
    label_template = """The following is a summary of a clustered dataset providing the topic names, growth (indicating the change in publication output, positive figures mean the discipline was growing) and average citations received per article published. Second is the same data but for either a single title published within the subject category or publications from a single country or region. Both datasets are separated by a "//". Please compare the two tables, indicating how the comparator compares to the overall subject category, furthermore suggest what topics the comparator should target to publish more papers and gain more citations. Use numbers and percentages in your summary and provide as much detail as possible in your response.
        {text}
    """
    
    # Split the combined string into smaller chunks, if needed
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(search_string)
    
    # Prepare documents for OpenAI model processing
    docs = [Document(page_content=t) for t in texts]
    
    # Define the prompt using the earlier template
    PROMPT = PromptTemplate(template=label_template, input_variables=["text"])
    
    # Load the summarization chain and run it on the prepared documents
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    output = chain.run(docs)
    
    # Split the output into individual sentences, while avoiding splitting on decimal points
    _list = re.split(r'(?<!\d)\.(?!\d|$)', output.replace("\n", ""))
    _list = [i.strip() for i in _list]
    
    return _list

def comparator_summary(topic_df, comparator_df):
    """
    Generates a comparative summary between a general topic dataset and a comparator dataset.
    
    This function first generates summarized tables for both datasets. It then converts these tables 
    to formatted strings. These strings are passed to the `comparator_analysis` function, which uses 
    OpenAI's model to produce a comparative analysis. The resulting analysis is processed to ensure 
    each statement ends with a period.
    
    Parameters:
    - topic_df (pandas.DataFrame): DataFrame representing the general topic dataset.
    - comparator_df (pandas.DataFrame): DataFrame representing the comparator dataset.
    
    Returns:
    - list: A list of sentences forming the comparative analysis between the topic and comparator datasets.
    
    Notes:
    - It's assumed that both DataFrames have the same structure suitable for the `generate_table_summary` function.
    """
    
    # Generate summarized tables for the topic and comparator datasets
    topic_table = generate_table_summary(topic_df)
    comparator_table = generate_table_summary(comparator_df)
    
    # Convert the summarized tables to formatted strings
    topic_df_string = get_df_string(topic_table)
    comparator_df_string = get_df_string(comparator_table)
    
    # Produce a comparative analysis using the OpenAI model
    summary = comparator_analysis(topic_df_string, comparator_df_string)
    
    # Ensure each statement in the analysis ends with a period
    summary = [i + "." for i in summary if i[-1] != "."]
    
    return summary

@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def generate_label(article_titles):
    """
    Generates a label for a set of research articles based on their titles using the OpenAI GPT-3.5-turbo model.
    
    Given a list of article titles, this function queries the OpenAI GPT-3.5-turbo model to produce a
    descriptive label that encapsulates the combined research topics. The label is then cleaned to remove
    any non-alphabetical characters.
    
    If a RateLimitError occurs when interacting with the OpenAI API, the function will exponentially back off 
    and retry until a maximum time of 60 seconds is reached.
    
    Parameters:
    - article_titles (list of str): List of article titles for which a combined label is to be generated.
    
    Returns:
    - str: A descriptive label for the combined research described by the provided article titles.
    
    Note:
    - Ensure that the OpenAI API key is properly configured before calling this function.
    - The function expects 'article_titles' to be in a format suitable for direct inclusion in the query string.
    """
    
    # Make a request to the OpenAI model to get a descriptive label
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=1,
        messages=[
            {
                "role": "user",
                "content": f"Please assign a topic label that adaquately represents the field of study of the following articles, please be as specific as possible, and return a label with no more than five words: {article_titles}",
            }
        ],
    )
    
    # Extract the label from the model's response and remove any newline characters
    label = completion.choices[0].message.content.replace("\n", "")
    
    # Clean the label by removing any non-alphabetical characters and return
    return re.sub(r'[^a-zA-Z ]', '', label)