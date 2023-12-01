import os
from typing import List
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from pathlib import Path
from IPython.display import Markdown, display, HTML
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import openai
import streamlit as st

from langchain.chat_models import ChatOpenAI
from semantic_search import SemanticSearch



def count_tokens(text: str) -> int:
    """Helper function to count tokens in a string"""
    # Note this is hard coded for OpenAI gpt models
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    text = str(text)
    return len(encoding.encode(text))


def add_month_year_to_df(df: pd.DataFrame, date_col_name: str) -> pd.DataFrame:
    """
    Create 'Jan 2023' formatted date. Used for printing citations.
    
    Example: df = add_month_year_to_df(df, 'coverage_date_completed')
    
    Args:
        df (pd.DataFrame): Dataframe containing a datetime column.
        date_col_name (str): The column containing the date.
    Returns:
        pd.DataFrame: Input df with a new column named 'datestamp'.
    """
    df[date_col_name] = pd.to_datetime(df[date_col_name])
    df['datestamp'] = df[date_col_name].dt.to_period('M')
    df['datestamp'] = df['datestamp'].dt.strftime('%b %Y')
    df['datestamp'] = df['datestamp'].astype('string')
    return df


def extract_citation_numbers_in_brackets(text: str) -> List[str]:
    """
    Captures citations from LLM response. 
    The target format is an integer inside of square brackets, i.e., [1].
    
    Example: citation_numbers = extract_citation_numbers_in_brackets(response)
    
    Args:
        text (str): LLM 'response' output string.
    Returns:
        List[str]: A list containing citations used by the LLM. 
    """
    matches = re.findall(r'\[(\d+)\]', text)
    return list(set(matches))


def print_cited_sources(df: pd.DataFrame, citation_numbers: List[str]) -> None:
    """
    Display cited sources with active links.
    Note: This is only a temporary solution to test if we can get proper hyperlinks from LLM output.
    
    """
    for citation in citation_numbers:
        i = int(citation) - 1  # convert string to int and adjust for 0-indexing
        title = df.iloc[i]["llm_title"]
        link = f"{df.iloc[i]['full_link']}"
        venue = df.iloc[i]["State"]
        date = df.iloc[i]["datestamp"]
        number = df.iloc[i]["index"]
        print(f"{[i+1]} [{title}]({link}) - {venue}, {date}, Number: {number}\n\n")



def get_llm_fact_pattern_summary(df: pd.DataFrame, text_col_name: str) -> pd.DataFrame:
    """
    Function for using an LLM to prep top n search results.
    Operates on one row at a time so that exceptions can be caught in a way that doesn't break the 'chain'
    
    Example: top_n_res_df = get_llm_fact_pattern_summary(df=top_n_res_df, text_col_name="situation_question_clean")
    
    Args:
        df (pd.DataFrame): Results from top n search.
        text_col_name (str): The column of text to process. Should already be pre-processed.
    Returns:
        pd.DataFrame: Input df with a new column named 'summary'.
    """
    
    llm_text_prep_system_template = "You are an actuarial documentation specialist. You're helping me summarize legal questions related to insurance."
    llm_text_prep_human_template = "Using the following scenario, write a comprehensive fact-based summary that focuses on insurance and legal topics.\n\nDESCRIPTIONS:{text}\n\nSUMMARY:"


    llm_text_prep_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(llm_text_prep_system_template),
            HumanMessagePromptTemplate.from_template(llm_text_prep_human_template),
        ],
        input_variables=["text"],
    )

    llm_text_prep_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=512), prompt=llm_text_prep_prompt, verbose=False
    )
    
    df = df.reset_index(drop=False)
    list_to_chain = df[[text_col_name]].to_dict('records')
    
    batch_size = 1
    results = []
    for i, batch_start in tqdm(enumerate(range(0, len(list_to_chain), batch_size)), total=len(list_to_chain)//batch_size):
        try:
            batch = list_to_chain[batch_start: batch_start + batch_size]
            processed_batch = llm_text_prep_chain.run(batch)
            results.append(processed_batch)
        except (openai.error.InvalidRequestError, ValueError) as e:
            results.append({'text': str(e)})
    assert len(results) == len(df), "Length of the result does not match the original data!"
    
    df['summary'] = results
    df['summary'] = df['summary'].astype('string')
    return df



def rerank_with_cross_encoder(df: pd.DataFrame,
                              query: str, 
                              text_col_name: str = "summary",
                              model_name: str = 'BAAI/bge-reranker-large',
    ) -> pd.DataFrame:
    """
    A function to rerank search results using pre-trained cross-encoder
    
    On models:
    Base models are listed below. More info here: https://www.sbert.net/docs/pretrained-models/ce-msmarco.html
    
    Example: rerank_res_df = rerank_with_cross_encoder(top_n_res_df, query, 'summary')
    
    Args:
        df (pd.DataFrame): Results from `get_llm_fact_pattern_summary`.
        text_col_name (str): The column of text to embed. Defaults to "summary". 
    Returns:
        pd.DataFrame: Input df sorted based on Instructor embeddings re-ranking.
    """
    # Initialize the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Prepare the data for the model
    query_df = pd.DataFrame({"query": [query]})
    query_fact_pattern_df = get_llm_fact_pattern_summary(df=query_df, text_col_name="query")
    query_proc = query_fact_pattern_df["summary"].iloc[0]  
    data = [(query_proc, text) for text in df[text_col_name].tolist()]
    # Tokenize the data
    features = tokenizer(*zip(*data), padding=True, truncation=True, return_tensors="pt")
    # Predict the scores
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
    # Convert scores to numpy array
    scores = scores.detach().numpy()
    # Add the scores to the dataframe
    df['scores'] = scores
    # Sort the dataframe by the scores in descending order
    df = df.sort_values(by='scores', ascending=False)
    return df



# Create context for LLM prompt
def create_context(df: pd.DataFrame, query: str, context_token_limit: int = 3000) -> str:
    """
    Function to create context for LLM prompt. 
    Designed to take in results from `rerank`, and add max number of search results based on a specified token limit.
    
    Args:
        df (pd.DataFrame): Results from `rerank`.
        query (str): The target test for the search. Should be pre-processed.
        token_limit (int): Max tokens to add to the context. 
            Defaults to 3k, which allows for ~1k output tokens (if using model with ~4k context).
    Returns:
        str: Formatted LLM context.
    """
    df.reset_index(drop=True, inplace=True)
    returns = []
    count = 1
    total_tokens = count_tokens(query)  # Start with tokens from the query
    # Add the text to the context until the context is too long
    for i, row in df.iterrows():
        # row['full_link'] = f"{row['full_link'].replace(' ', '')}"
        text = ("["
                + str(count)
                + "] "
                + row["summary"]
                + "\nURL: "
                + row["full_link"])
        text_tokens = count_tokens(text)
        if total_tokens + text_tokens > context_token_limit:
            break
        returns.append(text)
        total_tokens += text_tokens
        count += 1
    return "\n\n".join(returns)



def create_formatted_input(df: pd.DataFrame, query: str, context_token_limit: int,
    instructions: str ="""Instructions: Using the provided search results, and starting with the most relevant, write a detailed comparative analysis for a new query. ALWAYS cite results using [[number](URL)] notation after the reference. \
        End with a markdown table explaining ALL of the search results titled 'Top N Most Similar Cases' that includes 1) Citation Number, 2) Similarity Score, and 3) Legal Questions, where similarity score is a 1 to 10 score of how similar the text is to the new query.\n\nNew Query:""",
) -> str:
    """
    Creates formatted prompt combining top n search results (context), the target query, and final instructions for the LLM.
    
    Example: formatted_input = create_formatted_input(rerank_res_df, query)
    
    Args:
        df (pd.DataFrame): Results from `rerank`.
        query (str): The target test for the search. Should be pre-processed.
    Returns:
        str: Formatted LLM prompt.
    """

    context = create_context(df, query, context_token_limit)

    try:
        prompt = f"""{context}\n\n{instructions}\n{query}\n\nAnalysis:"""
        return prompt
    except Exception as e:
        print(e)
        return ""
    
    
def get_final_answer(formatted_input: str, llm) -> str:
    
    main_system_template = "You are helpful legal research assistant. Analyze the current legal question, and compare it to past cases. Using only the provided context, offer insights into how the researcher can reference the past questions to address the new outstanding issue. Do not answer the question or provide opinions, only draw helpful comparisons. Remember to use markdown links when citing the context, for example [[number](URL)]."
    main_human_template = "Search Results:\n\n{text}"

    main_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(main_system_template),
            HumanMessagePromptTemplate.from_template(main_human_template),
        ],
        input_variables=["text"],
    )
    
    answer_chain_gpt = LLMChain(
        llm=llm, prompt=main_prompt, verbose=False)
    
    response = answer_chain_gpt.run({"text": formatted_input})  
    return response


def print_results(df: pd.DataFrame, query: str, response: str) -> None:
    """Helper that prints everything"""
    citation_numbers = extract_citation_numbers_in_brackets(response)
    print(f"## New Query:")
    print(f"{query}")
    # print(f"## Model Response:")
    print(f"{response}")
    print_cited_sources(df, citation_numbers)
    

def get_df():
    """Returns a pandas DataFrame."""
    df = pd.read_parquet("reddit_legal_cluster_test_results.parquet")
    df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')
    df['datestamp'] = df['timestamp'].dt.date
    return df


def get_llm(temperature: float = 0, model: str = "gpt-3.5-turbo"):
    llm = ChatOpenAI(temperature=temperature, model=model)
    return llm


# Main function
MODEL_NAME = "gpt-3.5-turbo"
CONTEXT_TOKEN_LIMIT = 3000

def run_tool(user_query, top_n=5, model_name=MODEL_NAME, context_token_limit=CONTEXT_TOKEN_LIMIT):
    """
    Function to run the entire process end-to-end.
    
    Args:
        user_query (str): The user's text query.
        model_name (str): The name of the model to use.
        token_limit (int): The maximum number of tokens for the context.
    Returns:
        str: A string containing the user query, model response, and cited sources.
    """
    # Read in a df
    df = get_df()
    load_dotenv()
    df = get_df()
    search_engine = SemanticSearch(df)
    llm = get_llm(model=model_name)

    # Create instance of SemanticSearch
    search_engine = SemanticSearch(df)

    # Query top n
    top_n_res_df = search_engine.query_similar_documents(
        user_query,
        top_n = top_n,
        filter_criteria = None,
        use_cosine_similarity = True,
        similarity_threshold = 0.93)

    # Run get_llm_fact_pattern_summary
    try:
        top_n_res_df = get_llm_fact_pattern_summary(df=top_n_res_df, text_col_name="body")
    except Exception as e:
        print(f"Error in get_llm_fact_pattern_summary: {e}")
        return

    # Run rerank_with_cross_encoder
    try:
        rerank_res_df = rerank_with_cross_encoder(top_n_res_df, user_query, 'summary')
    except Exception as e:
        print(f"Error in rerank: {e}")
        return

    # Run create_formatted_input
    try:
        formatted_input = create_formatted_input(rerank_res_df, user_query, context_token_limit=context_token_limit)
    except Exception as e:
        print(f"Error in create_formatted_input: {e}")
        return

    # Run get_final_answer
    try:
        response = get_final_answer(formatted_input, llm)
    except Exception as e:
        print(f"Error in get_final_answer: {e}")
        return
    
    # Create a string containing the user query, model response, and cited sources
    # result = f"## New Query:\n{user_query}\n## Model Response:\n{response}\n"
    result = f"\n{response}\n"
    citation_numbers = extract_citation_numbers_in_brackets(response)
    for citation in citation_numbers:
        i = int(citation) - 1  # convert string to int and adjust for 0-indexing
        title = rerank_res_df.iloc[i]["llm_title"]
        link = f"{rerank_res_df.iloc[i]['full_link']}"
        venue = rerank_res_df.iloc[i]["state"]
        date = rerank_res_df.iloc[i]["datestamp"]
        number = rerank_res_df.iloc[i]["index"]
        result += f"<br><b>{[i+1]} [{title}]({link}) - {venue}, {date}, Number: {number}</b>"

    return result
    
if __name__=="main":
    run_tool()