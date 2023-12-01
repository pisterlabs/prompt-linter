import openai
from termcolor import colored
import pandas as pd
import re
import tiktoken
import os
import requests
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
# import streamlit as st
# from .icici_chat import construct_prompt
# from database import get_redis_connection,get_redis_results
from dotenv import load_dotenv

load_dotenv()

# Openai API KEY for model
API_KEY = os.getenv("AZURE_OPENAI_API_KEY") 
RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") 

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
# openai.api_version = "2022-12-01"
openai.api_version = "2023-07-01-preview"

url = openai.api_base + "/openai/deployments?api-version=2022-12-01" 

r = requests.get(url, headers={"api-key": openai.api_key})


# embedding model
EMBEDDING_MODEL = "bob-text-embedding-ada-002"
# A basic class to create a message as a dict for chat
openai.api_key = API_KEY
# Read icici data for embedding
df = pd.read_csv('Bob_chunks.csv')


# df["token"] = None
# for idx, r in df.iterrows():
# #     print(len(r.content))
# #     df["token"] = df[len(r.content)]
#     df.loc[idx,'token'] = len(r.content)

# function convert text data into embed or vector format
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        engine=EMBEDDING_MODEL,
        input=text
    )
    return result["data"][0]["embedding"]


# import time
# def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
#     embeddings = {}
#     for idx, r in df.iterrows():
#         print(idx)
# #         print(r)
#         embeddings[idx] = get_embedding(r.title)
#         time.sleep(5)  # Add a delay of 10 second between requests
#     return embeddings

# function load the embedded data
def load_embeddings(fname: "str") -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)

    # max_dim = max([int(c) for c in df.columns if c != "title"])
    # max_dim = max([int(c) for c in df.columns if c != "title"])
    max_dim = 0
    for c in df.columns:
        if c != "title" and not c.startswith("version"):
            try:
                max_dim = max(max_dim, int(c))
            except ValueError:
                pass

    return {
        (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }


document_embeddings = load_embeddings("Bob_embedded.csv")


# function used for similarity find from the vector data or embedded data
def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


# function used for order the similarity
def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[
    (float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (cosine_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities


# a = order_document_sections_by_query_similarity("Give me a list of icici credit crads  interest rate", document_embeddings)[:5]

# max length used for token limit set
MAX_SECTION_LEN = 7552

SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

print(f"Context separator contains {separator_len} tokens")


# function used for the prompt construction
def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)[:5]
    #     print(most_relevant_document_sections)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        #         print(section_index)
        document_section = df.loc[section_index]
        #         print(document_section)
        chosen_sections_len += document_section.token + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        # choose section to append the data lower than max section length
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
        context = "".join(chosen_sections)
    # Useful diagnostic information
    # print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    # print(type("".join(chosen_sections) ))

    # header used for the data come in the good format

    # header = """Answer the question as truthfully as possible using the provided Given Below Information and previous conversation, and if the answer is not contained within in both, say "I don't know.For more details call on our CUSTOMER CARE NO.1800 1080"\n\nGiven Below Information:"""
    # header = """Answer the question as truthfully as possible using the provided Given Below Information only,\n And if the answer is not contained within the text below, say "I don't have that information.For more details call on our CUSTOMER CARE NO.1800 1080 or visit our website https://www.icicibank.com/ and Don't justify your answers. Don't give information not mentioned in\nDon't make any change in urls\nif there any urls like website and image in Given Data must give it with response \n Given Below Information:\n"""
    header = f"""You are a friendly, conversational bank of baroda(BOB) assistant.You can use/refer the following context whats available urls, image urls,data, help find what they want, and answer questions.
It's ok if you don't know the answer simply say "i don't have that information".

Context:\"""
{context}
\"""
To get the best response, please keep the following guidelines in mind:

Refrain from sharing any company/bank and product details and names and urls/links outside of Bank of baroda(bob).
Sharing other companies/bank Details , URLs, Information of other Company can damage the reputation of Company so keep it on mind and generate response
Please note that you should not provide direct comparisons between Bank of baroda(BOB) and  other Banks. Focus solely on providing information about bank of baroda services.Focus solely on providing information about bank of baroda(BOB) services.
You can provide useful URLs and image URLs based on the given context in your response.
When answering your questions, you will present the information in bullet points or Tabular formate.
Please note that Yoy won't justify my responses.
Provide a concise description of what makes Bank of baroda(bob) unique or highlight their key offerings. Engage the user's interest and set the context
You can find more information on Bank of Baroda(BOB) official website "https://www.bankofbaroda.in/". This Url must presernt in every response

Question:\"
{question}

Ensure that you have follow all above guidelines to make response greate to user.Create Response To increase bank of baroda(bob) reputaion to customer.

\"""

Helpful Answer:"""
    print(header)
    return header


# conversation_history = []

# get most relevant data from the vectorized data
def get_search_results(prompt):
    latest_question = prompt
    search_content = construct_prompt(latest_question, document_embeddings, df)
    # print("SEARCH    ",search_content)
    # get_redis_results(redis_client,latest_question,INDEX_NAME)['result'][0]

    return search_content


def ask_assistent(prompt,conversation):
        conversation_history = []
        # system = get_search_results(query1)
        prompt='''Your task is to transform a input question from a conversation history into a Rephrase question. The transformed question should be independent of the preceding discussion but can adapt to relevant features from the conversation. If the question is related to the previous conversation, understand what user want to know and replace that keyword from conversation history. If it is not related or same, still formulate a question from the input. The model should accurately identify the language in which the user asked the question and translate it into English while maintaining the original context. Append the phrase "Give the answer in [language phase of Input Question]" to the translated question.'''+'''
        Conversation History:
        {conversation}

        Input Question:
        {prompt}

        Rephrase Question:'''.format(prompt=prompt, conversation=conversation)
        message = {"role": "system", "content": "You are generate question"}  
        conversation_history.append(message)
        user = {"role": "user", "content": prompt}
        conversation_history.append(user)
        # conversation.append({"content": query1})
        
        response = openai.ChatCompletion.create(
            engine="bob-gpt-35-turbo",
            messages = conversation_history,
            
            
        )
        print("\n" + response['choices'][0]['message']['content'] + "\n")       
        text = response['choices'][0]['message']['content']
        return text
    
       
def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        conversation:list

) -> str:

     
    
    conversation_history = []
    myMessages = []
    # Send prompt to OpenAI API and get response
    while(True):
        # user_input = input("")  
        query1 = ask_assistent(query,conversation)
        system = get_search_results(query1)
       
        myMessages.append({"role": "system", "content": "You are expert bank of baroda(BOB) advisor"})

    # myMessages.append({"role": "user", "content": "context:\n\n{}.\n\n Answer the following user query according to above given context:\nuser_input: {}".format(context,user_input)})
        myMessages.append({"role": "user",
                       "content": f"{system}"})
        
        
        # conversation.append({"content": query1})
        
        response = openai.ChatCompletion.create(
            engine="bob-gpt-35-turbo",
            messages = myMessages,
            # temperature=0.3,
            stream = True,
            
        )
        
        return response 
        
        
def inputdata(inpval: str,conversation:list) :
    response = answer_query_with_context(inpval, df,conversation)
    print(response)
    return response