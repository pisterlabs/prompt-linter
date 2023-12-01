import arxiv
import ast
import concurrent
from csv import writer
import openai
import os
import pandas as pd
from PyPDF2 import PdfReader
import requests
from scipy import spatial
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from tqdm import tqdm
from termcolor import colored
import json
from IPython.display import display, Markdown

GPT_MODEL = "gpt-3.5-turbo-0613"
EMBEDDING_MODEL = "text-embedding-ada-002"

# defining the api key from environment variable or inserting it directly
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = "sk-<your key here>"


# Importing necessary libraries
from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests

# Defining a function that sends a request to the OpenAI API for chat completion
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, model=GPT_MODEL):
    # Setting headers for the request
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    # Creating a JSON object with the request data
    json_data = {"model": model, "messages": messages}
    # Adding functions to the JSON object if they are provided
    if functions is not None:
        json_data.update({"functions": functions})
    try:
        # Sending a POST request to the OpenAI API with the JSON object as the payload
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        # If there is an exception, print an error message and return the exception
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


# Defining a Conversation class
class Conversation:
    # Initializing the conversation history as an empty list
    def __init__(self):
        self.conversation_history = []

    # Defining a method to add a message to the conversation history
    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    # Defining a method to display the conversation history
    def display_conversation(self):
        # Defining a dictionary to map roles to colors for display
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        # Iterating through the conversation history and displaying each message with its role and color
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )

'''Downloaded papers will be stored in a directory (we use ./data/papers here). We create a file arxiv_library.csv to store the embeddings and details for downloaded papers to retrieve against using summarize_text.'''
# Set a directory to store downloaded papers
data_dir = os.path.join(os.curdir, "data", "papers")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
paper_dir_filepath = "./data/arxiv_library.csv"

# Generate a blank dataframe where we can store downloaded files
df = pd.DataFrame(list())
df.to_csv(paper_dir_filepath)


# Defining a function that sends a request to the OpenAI API for text embedding
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def embedding_request(text):
    response = openai.Embedding.create(input=text, model=EMBEDDING_MODEL)
    return response


def get_articles(query, library=paper_dir_filepath, top_k=5):
    """This function gets the top_k articles based on a user's query, sorted by relevance.
    It also downloads the files and stores them in arxiv_library.csv to be retrieved by the read_article_and_summarize.
    """
    # Search for articles based on the user's query
    search = arxiv.Search(
        query=query, max_results=top_k, sort_by=arxiv.SortCriterion.Relevance
    )
    result_list = []
    for result in search.results():
        result_dict = {}
        # Add the title and summary of the article to the result dictionary
        result_dict.update({"title": result.title})
        result_dict.update({"summary": result.summary})

        # Taking the first url provided
        # Add the article URL and PDF URL to the result dictionary
        result_dict.update({"article_url": [x.href for x in result.links][0]})
        result_dict.update({"pdf_url": [x.href for x in result.links][1]})
        result_list.append(result_dict)

        # Store references in library file
        # Get the text embedding for the article title
        response = embedding_request(text=result.title)
        # Create a list with the article title, downloaded PDF file path, and text embedding
        file_reference = [
            result.title,
            result.download_pdf(data_dir),
            response["data"][0]["embedding"],
        ]

        # Write the file reference to the library file
        with open(library, "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(file_reference)
            f_object.close()
    return result_list

# Test that the search is working
# result_output = get_articles("ppo reinforcement learning")
# print(result_output[0])

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
) -> list[str]:
    """
    Returns a list of strings and relatednesses, sorted from most related to least.
    
    Args:
    - query (str): the query string to compare against the embeddings in the dataframe
    - df (pd.DataFrame): the dataframe containing the embeddings to compare against
    - relatedness_fn (function): the function to use to calculate relatedness between embeddings
    - top_n (int): the number of top results to return
    
    Returns:
    - list[str]: a list of strings sorted by relatedness to the query string
    """
    # Get the embedding for the query string
    query_embedding_response = embedding_request(query)
    query_embedding = query_embedding_response["data"][0]["embedding"]
    
    # Calculate the relatedness between the query embedding and each embedding in the dataframe
    strings_and_relatednesses = [
        (row["filepath"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]    

    # Sort the strings by relatedness and return the top n results
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n]

def read_pdf(filepath):
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a pdf reader object
    reader = PdfReader(filepath)
    pdf_text = ""
    page_number = 0
    for page in reader.pages:
        page_number += 1
        pdf_text += page.extract_text() + f"\nPage Number: {page_number}"
    return pdf_text

# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    """Returns successive n-sized chunks from provided text."""
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j

def extract_chunk(content, template_prompt):
    """This function applies a prompt to some input content. In this case it returns a summarize chunk of text"""
    prompt = template_prompt + content
    response = openai.ChatCompletion.create(
        model=GPT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return response["choices"][0]["message"]["content"]

# This function summarizes text from an academic paper
def summarize_text(query):
    """This function does the following:
    - Reads in the arxiv_library.csv file in including the embeddings
    - Finds the closest file to the user's query
    - Scrapes the text out of the file and chunks it
    - Summarizes each chunk in parallel
    - Does one final summary and returns this to the user"""

    # A prompt to dictate how the recursive summarizations should approach the input paper
    summary_prompt = """Summarize this text from an academic paper. Extract any key points with reasoning.\n\nContent:"""

    # If the library is empty (no searches have been performed yet), we perform one and download the results
    library_df = pd.read_csv(paper_dir_filepath).reset_index()
    if len(library_df) == 0:
        print("No papers searched yet, downloading first.")
        get_articles(query)
        print("Papers downloaded, continuing")
        library_df = pd.read_csv(paper_dir_filepath).reset_index()

    # Rename the columns of the dataframe
    library_df.columns = ["title", "filepath", "embedding"]
    # Convert the embeddings column from string to list
    library_df["embedding"] = library_df["embedding"].apply(ast.literal_eval)
    # Find the closest file to the user's query
    strings = strings_ranked_by_relatedness(query, library_df, top_n=1)
    print("Chunking text from paper")
    # Read the text from the file
    pdf_text = read_pdf(strings[0])

    # Initialise tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    results = ""

    # Chunk up the document into 1500 token chunks
    chunks = create_chunks(pdf_text, 1500, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    print("Summarizing each chunk of text")

    # Parallel process the summaries
    # Create a ThreadPoolExecutor with a maximum number of workers equal to the number of text chunks
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(text_chunks)
    ) as executor:
        # Submit a task to the executor for each text chunk, using the extract_chunk function and summary_prompt as arguments
        futures = [
            executor.submit(extract_chunk, chunk, summary_prompt)
            for chunk in text_chunks
        ]
        # Create a progress bar with a total number of steps equal to the number of text chunks
        with tqdm(total=len(text_chunks)) as pbar:
            # Wait for each task to complete and update the progress bar
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        # Concatenate the results from each task into a single string
        for future in futures:
            data = future.result()
            results += data

    # Final summary
    print("Summarizing into overall summary")
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Write a summary collated from this collection of key points extracted from an academic paper.
                        The summary should highlight the core argument, conclusions and evidence, and answer the user's query.
                        User query: {query}
                        The summary should be structured in bulleted lists following the headings Core Argument, Evidence, and Conclusions.
                        Key points:\n{results}\nSummary:\n""",
            }
        ],
        temperature=0,
    )
    return response

# Test the summarize_text function works
# chat_test_response = summarize_text("PPO reinforcement learning sequence generation")
# print(chat_test_response["choices"][0]["message"]["content"])

# Initiate our get_articles and read_article_and_summarize functions
arxiv_functions = [
    {
        "name": "get_articles",
        "description": """Use this function to get academic papers from arXiv to answer user questions.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            User query in JSON. Responses should be summarized and should include the article URL reference
                            """,
                }
            },
            "required": ["query"],
        },
        "name": "read_article_and_summarize",
        "description": """Use this function to read whole papers and provide a summary for users.
        You should NEVER call this function before get_articles has been called in the conversation.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            Description of the article in plain text based on the user's query
                            """,
                }
            },
            "required": ["query"],
        },
    }
]

def chat_completion_with_function_execution(messages, functions=[None]):
    """This function makes a ChatCompletion API call with the option of adding functions"""
    response = chat_completion_request(messages, functions)
    full_message = response.json()["choices"][0]
    if full_message["finish_reason"] == "function_call":
        print(f"Function generation requested, calling function")
        return call_arxiv_function(messages, full_message)
    else:
        print(f"Function not required, responding to user")
        return response.json()


def call_arxiv_function(messages, full_message):
    """Function calling function which executes function calls when the model believes it is necessary.
    Currently extended by adding clauses to this if statement."""

    if full_message["message"]["function_call"]["name"] == "get_articles":
        try:
            parsed_output = json.loads(
                full_message["message"]["function_call"]["arguments"]
            )
            print("Getting search results")
            results = get_articles(parsed_output["query"])
        except Exception as e:
            print(parsed_output)
            print(f"Function execution failed")
            print(f"Error message: {e}")
        messages.append(
            {
                "role": "function",
                "name": full_message["message"]["function_call"]["name"],
                "content": str(results),
            }
        )
        try:
            print("Got search results, summarizing content")
            response = chat_completion_request(messages)
            return response.json()
        except Exception as e:
            print(type(e))
            raise Exception("Function chat request failed")

    elif (
        full_message["message"]["function_call"]["name"] == "read_article_and_summarize"
    ):
        parsed_output = json.loads(
            full_message["message"]["function_call"]["arguments"]
        )
        print("Finding and reading paper")
        summary = summarize_text(parsed_output["query"])
        return summary

    else:
        raise Exception("Function does not exist and cannot be called")
    
# Start with a system message
paper_system_message = """You are arXivGPT, a helpful assistant pulls academic papers to answer user questions.
You summarize the papers clearly so the customer can decide which to read to answer their question.
You always provide the article_url and title so the user can understand the name of the paper and click through to access it.
Begin!"""
paper_conversation = Conversation()
paper_conversation.add_message("system", paper_system_message)

# Add a user message
paper_conversation.add_message("user", "Hi, how does PPO reinforcement learning work?")
chat_response = chat_completion_with_function_execution(
    paper_conversation.conversation_history, functions=arxiv_functions
)
assistant_message = chat_response["choices"][0]["message"]["content"]
paper_conversation.add_message("assistant", assistant_message)
print(assistant_message)
