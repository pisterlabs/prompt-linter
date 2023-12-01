import os
from langchain.embeddings import OpenAIEmbeddings
import langchain
from annoy import AnnoyIndex
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer, util
import sys
from langchain.document_loaders import PyPDFLoader
from flask import Flask, request, render_template, jsonify
app = Flask(__name__)

embeddings = OpenAIEmbeddings(openai_api_key="OPEN AI KEY")
model = SentenceTransformer('sentence-transformers/allenai-specter', device='cpu')

def get_embeddings_per_page(page):
    try:
        ret = embeddings.embed_query(page)
        return ret
    except:
        return None

policy_manual = "policy_manual.pdf"
loader = PyPDFLoader(policy_manual)
pages = loader.load_and_split()

print("There are " + str(len(pages)) + " pages in the policy manual.")

EMBEDDING_DIM = 1536
def get_embeddings_for_text(text):
    return embeddings.embed_query(text)
def load_index_map():
    index_map = {}
    with open('index_map' + name + '.txt', 'r') as f:
        for line in f:
            idx, path = line.strip().split('	')
            index_map[int(idx)] = path
    return index_map

def query_top_files(query, top_n=4):
    # Load annoy index and index map
    t = AnnoyIndex(EMBEDDING_DIM, 'angular')
    t.load(name+'_ada.ann')
    # Get embeddings for the query
    query_embedding = get_embeddings_for_text(query)
    # Search in the Annoy index
    indices, distances = t.get_nns_by_vector(query_embedding, top_n, include_distances=True)
    # Fetch file paths for these indices
    return (indices, distances)

def query_top_files_specter(query, top_n=4):
    # Load annoy index and index map
    t = AnnoyIndex(768, 'angular')
    t.load(name + '_specter.ann')
    # Get embeddings for the query
    query_embedding = model.encode(query)
    # Search in the Annoy index
    indices, distances = t.get_nns_by_vector(query_embedding, top_n, include_distances=True)
    # Fetch file paths for these indices
    return (indices, distances)

name = "PolicyManual"
llm = OpenAI(openai_api_key="OPEN AI KEY", model_name= "gpt-3.5-turbo-16k")


@app.route('/')
def my_form():
    return render_template('myform.html')

@app.route('/', methods=['POST'])
def search_endpoint():
    ##args=request.args
    query = request.form['text']
    template = """ Take a deep breath.
                    extract the important phrases from the query {query} and output the phrase"""
    prompt = PromptTemplate(template=template, input_variables=["query"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    phrase = ""
    try:
        phrase = llm_chain.run({'query': query})
        print (phrase)
    except Exception as e:
        print (e)
    (indices, distances) = query_top_files(query, 10)
    (indices2, distances2) = query_top_files_specter(query, 5)
    (indices3, distances3) = query_top_files(phrase, 5)
    indices = indices + indices3 + indices2

    response = ""
    for ind in indices:
        response += "PAGE_NUMBER:" + str(ind) + "\n"
        response +=  "CONTENT:" + pages[ind].page_content + "\n"
        response += "----------------------------------\n"

    template = """ Take a deep breath. 
    You are an AI Assitant helping with answering questions from USCIS field manual.

    TASK DESCRIPTION: 
    User wants to get answer for the QUESTION QUERY {query} using the USCIS field manual.
    Below is the USCIS field manual pages relevant to the query. It is organized as PAGE_NUMBER followed by CONTENT {response}
    PROVIDE an answer in HTML FORMAT to the QUESTION QUERY based on the USCIS field manual pages meeting the HARD REQUIREMENTS below
    Key Phrases in the QUESTION QUERY: {phrase}

    HARD REQUIREMENTS
    1. The answer should be STRICTLY BASED ON the USCIS field manual pages provided above
    2. The answer should be in plain English understandable by a human
    3. Quote the  PAGE NUMBER AND PHRASES from the USCIS field manual pages provided above in the answer with Bold font
    4. The answer should be in HTML FORMAT and should be in the form of BULLET POINTS if possible
    5. Recommend a follow up to the user to ask based on the answer provided
    6. Highlight the KEY PHRASES in the answer with BOLD <b> tags
    7. The answer should be READABLE with emojis and punctuations
    
    """
    prompt = PromptTemplate(template=template, input_variables=["query",  "response", "phrase"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    try:
        output = llm_chain.run({'query': query, 'response': response, 'phrase': phrase})
        print (output)
        output = "<b> query:" + query + "</b><br>" + output
        output= output.replace("\n", "<br>")

        tokens = phrase.split()
        output = output.replace(query, "<b>" + query + "</b>")
        for token in tokens:
            token = token.replace("(", "")
            if (len(token) <= 3):
                continue
            output = output.replace(token, "<b>" + token + "</b>")
            output = output.replace("Page", "<b>PAGE</b>")
        output = output.replace("Follow up", "<br><b>Follow up</b><br>")

        output = output + " <br><br> I am powered by an LLM call which is known to HALLUCINATE. This is not a LEGAL advice and simply a proof of concept that LLMs can read through legal documents and fetch the necessary information."
    except Exception as e:
        print (e)

    return output
    
