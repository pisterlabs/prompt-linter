
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseRetriever, Document, HumanMessage
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.utilities.arxiv import ArxivAPIWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from typing import List
import os
import sys
import asyncio
import urllib.request
import feedparser
import urllib.request
import urllib.parse
import time
import xml.etree.ElementTree as ET
from panel.layout import Column
import panel as pn




openai_api_key = "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"
os.environ["OPENAI_API_KEY"] = openai_api_key

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# llm = OpenAI(model= "text-davinci-003", temperature=0.75)
llm = OpenAI(temperature=0.75)

chat = ChatOpenAI(temperature=0.75)


def get_questions(prompt):
    response = chat([HumanMessage(content=prompt)])
    questions = [response.content.strip()]
    return questions

def get_answers(question, abstract):
    messages = [HumanMessage(content=question), HumanMessage(content=f"abstract: {abstract}")]
    response = chat(messages)
    answer = response.content.strip()
    return answer


def get_arxiv_results( search_query, start=0, max_results=20):
    base_url = 'http://export.arxiv.org/api/query?'
    query = f'search_query=all:{urllib.parse.quote(search_query)}&start={start}&max_results={max_results}'
    response = urllib.request.urlopen(base_url+query).read()
    feed = feedparser.parse(response)
    return feed


def parse_arxiv_results(feed):
    entries = feed.entries
    results = []
    for entry in entries:
        result = {
            "arxiv_id": entry.id.split('/abs/')[-1],
            "title": entry.title,
            "authors": entry.author,
            "abstract": entry.summary
        }
        results.append(result)
    return results



file_input = pn.widgets.FileInput(width=300)

prompt = pn.widgets.TextEditor(
value="", placeholder="Enter your questions here...", height=160, toolbar=False
)
run_button = pn.widgets.Button(name="Run!")

select_k = pn.widgets.IntSlider(
    name="Number of relevant chunks", start=1, end=5, step=1, value=2
)
select_chain_type = pn.widgets.RadioButtonGroup(
    name='Chain type', 
    options=['stuff', 'map_reduce', "refine", "map_rerank"]
)

widgets = pn.Row(
    pn.Column(prompt, run_button, margin=5),
    pn.Card(
        "Chain type:",
        pn.Column(select_chain_type, select_k),
        title="Advanced settings", margin=10
    ), width=600
)




def qa(file, query, chain_type, k):
    # load document
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    print(result['result'])
    return result


def main():
    search_query = input("Enter a query: ")
    start = 0
    total_results = 20
    results_per_iteration = 5
    wait_time = 3

    print('Searching arXiv for %s' % search_query)

    while start < total_results:
        feed = get_arxiv_results(search_query, start=start, max_results=results_per_iteration)
        results = parse_arxiv_results(feed)
        for result in results:
            print('arxiv-id: %s' % result['arxiv_id'])
            print('Title:  %s' % result['title'])
            print('First Author:  %s' % result['authors'])
            print()
        start += results_per_iteration

    

    selected_paper_id = input("Enter the arxiv-id of the article you want to read: ")


    abstract_url = f'http://export.arxiv.org/api/query?id_list={selected_paper_id}'
    abstract_response = urllib.request.urlopen(abstract_url).read()


    root = ET.fromstring(abstract_response)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    abstract_elements = root.findall(".//atom:summary", namespace) 
    if not abstract_elements:
        print("No abstract found for this paper")
        print(f'LINK: https://arxiv.org/abs/{selected_paper_id}')
        abstract = '...'
    else:
        abstract = abstract_elements[0].text
  

    doc = Document(page_content= abstract, content=abstract)
    retrieved_from = {"name": "arxiv", "url": f"https://arxiv.org/abs/{selected_paper_id}"}
    
    pdf_file = f'https://arxiv.org/pdf/{selected_paper_id}.pdf'

    while True:

        query = input("Enter a question or type 'exit': ")
        if query == 'exit':
            break
        result = qa(file=pdf_file, query=query, chain_type="map_rerank", k=2)
        


if __name__ == "__main__":
    main()



