


import os
import pprint

os.environ["OPENAI_API_KEY"] = ""
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI

from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import pandas as pd
import requests

from langchain.document_loaders import TextLoader


def request_chatgpt(prompt):
    # Define your API key
    api_key = ""

    # Define the API endpoint URL
    endpoint = "https://api.openai.com/v1/chat/completions"

    # Define the API request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Define the API request data
    data = {
        "model": "gpt-3.5-turbo-16k",
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}]
    }

    # Send the API request
    try:
        response = requests.post(endpoint, headers=headers, json=data, timeout=80)
    except Exception:
        return ""
    response_text = ""

    # Check the API response status code
    if response.status_code == 200:
        # Print the API response text
        response_text = response.json()['choices'][0]['message']['content']
        # print(response_text)
    else:
        # Print the API error message
        print(f"Request failed with status code: {response.status_code}")

    return response_text


def summarize(path):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text = " ".join([_.page_content for _ in documents])

    prompt = f"Summarize the most pressing issues outlined and the initiatives as well as the desired outcomes of the project proposal in triple brackets in 200 to 250 words:\n\n((({text})))"

    return request_chatgpt(prompt)


def suggest(summary):
    loader = TextLoader("solution_summaries.txt")
    documents = loader.load()

    # split the documents into chunks
    text_splitter = CharacterTextSplitter(separator="\n\n====================", chunk_size=500)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 32})
    # create a chain to answer questions
    from langchain.chat_models import ChatOpenAI
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0), chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    query = "Which five of the following [[[solutions]]] best answer the problem posed in the initial (((project proposal))), and why? Consider relevance of the solutions, geography, economics, and any other potential reasons when answering. Return the 5 top solutions ranked from best to worst, in the following format:\n{\"NAME\": \"solutionname\", \"CATEGORY\": \"solutioncategory\", \"RECOMMENDATION\": \"how to apply the solution to the specific initiatives mentioned\", \"OUTCOME\": \"the expected effects of the application of the solution in the context of the proposal\", \"ISSUES\": \"potential issues with the application of the solution to the problem or areas which this solution does not address\"}. Make sure to output each solution in json format!\n\n"
    query += f"((({summary})))"

    result = qa({"query": query})

    result_text = result["result"]
    print()

    import json
    ret = []
    for r in result_text.split("\n"):
        try:
            r = json.loads(r)
            ret.append(r)
            print(r)
        except:
            pass

    return ret


summary = summarize("data\CPDs\CPD Somalia.pdf")
print(summary)
results = suggest(summary)

suggest("We are looking for ways to use drones for help in agriculture.")
