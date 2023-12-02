import langchain
from langchain.llms import OpenAI
# from langchain.llms.claude import Claude
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import openai
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
import json
import pandas as pd
import api_keys
import os
from langchain.docstore.document import Document
import pandas as pd
import random
# import api_keys

import pickle
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

#anthropic
import anthropic
import PyPDF2
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

api_key = os.getenv("OPENAI_API_KEY")
def summarise(path_to_pdf = 'data/a_pro-innovation_approach_to_AI_regulation.pdf'):
    llm = OpenAI(temperature=0, openai_api_key=api_key) 
    # llm = Claude()

    # Extract text from PDF
    pdf_path = path_to_pdf
    text = extract_text(pdf_path)

    text_splitter = TokenTextSplitter(chunk_size= 4000 - 256, chunk_overlap=0)

    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

def fake_data():
    data = {
    'Headline': ['AI regulation: pro-innovation, responsible growth.', 'Event 1', 'Event 1', 'Event 1'],
    'Sentiment': [1, 1, -1, 3],
    'Date': ['2020-01-01', '2021-03-01', '2022-04-01', '2022-05-01'],
    'Relevance': [0.7, 0.4, 0.8, 0.95],
    'Color': ['blue', 'blue', 'red', 'green'],
    'Topic' :['Submissions', 'Submissions', 'Internal sources', 'Public'],
    'Summary': [
    ' The UK government has proposed a new AI regulatory framework that includes five values-focused cross-sectoral principles to guide regulator responses to AI risks and opportunities. This framework is designed to provide clarity and coherence to the AI regulatory landscape, build the evidence base, and ensure risks are identified and addressed while also supporting innovation. It includes a set of functions to support implementation of the framework, such as monitoring, assessment and feedback, and cross-sectoral risk assessment. The government has opened a consultation to receive feedback from stakeholders, and plans to establish a regulatory sandbox for AI and engage with the public to build trust in the technology.', 
    'something', 'else', 'else']
    }
    return pd.DataFrame(data)

def get_position(relevant):
    position = None
    openai.api_key = api_keys.OPENAI
    system="""
    You are an AI assistant that reads summaries of documents and determines the opinions of the minister referred to in the documents about the topics discussed in the documents.

    You must do this in less than 100 words.
    """

    prompt = ""

    for i, row in relevant.iterrows():
        prompt = prompt + "\n" + row['Summary']

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )

    print('done')

    position = response['choices'][0]['message']['content']

    # do something
    if position is None:
        position = 'We do not have any position on this submission at the moment.'
    return position

def doc_to_json(path):
    openai.api_key = api_keys.OPENAI
    system="""
    You are an AI assistant that converts document text into valid json in the following format:

    {
        "Type": <type of document, this can only be one of the following: Email, Meeting minutes, Speech, or Submission>,
        "Title": <title of the document>,
        "Topic": <overall topic discussed in the document>,
        "Summary": <summary of the document content>,
        "Content": null,
        "Relevant People": <list of people mentioned in the document, inlcuding sender and recipients, format as: Forname Surname>,
        "Director": <director mentioned in the document>,
        "Team": <teams mentioned in the document that need to give clearance, where a name is provided>,
        "Sentiment": <sentiment analysis of the document, should be a floating point number between -1.00 and 1.00, with -1.00 being negative and 1.00 being positive and 0.00 being neutral, must be filled>,
        "Deadline": <deadline of any actions mentioned in the document, format as exactly as shown in the document>,
        "Actions": <any follow-up actions mentioned in the document, with added deadline as per "Deadline" field>,
        "Notes": <any notes attached to the document>,
        "Date": <date of the document in the format: YYYY-MM-DD, if no date is found then null>
    }

    You must extract this information from the document accuractly.
    """

    prompt = extract_text(path)

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )

    print('done')

    return json.loads(response['choices'][0]['message']['content'])

def summary_with_claude(fine_name='regulations.pdf'):
    Anthropic(api_key=api_keys.API_CLAUDE)
    with open(fine_name, 'rb') as pdf_file:
        # Initialize a PDF file reader
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Initialize a string to hold all the text
        pdf_text = ''

        # Loop through each page in the PDF
        for page_num in range(len(pdf_reader.pages)):
            # Get the text from the page and add it to the total text
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()

    # Now `pdf_text` contains all the text from the PDF
    prompt = " '"+pdf_text+"' summarise this document "


    res = anthropic_client.completions.create(prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}", model="claude-v1.3-100k", max_tokens_to_sample=40000)

    return res.completion

def analyze_sentiment(text):
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can use other engines like "text-curie-003"
        prompt=f"The following text is: '{text}'. The sentiment of this text is",
        temperature=0.3,
        max_tokens=60
    )

    return response.choices[0].text.strip()

    # Test the sentiment analysis
    text = "I am very happy today!"
    sentiment = analyze_sentiment(text)
    print(f'The sentiment is: {sentiment}')

#data = []
#import os
#for filename in os.listdir('data'):
#    f = os.path.join('data', filename)
#    # checking if it is a file
#    if os.path.isfile(f):
#        data.append(json.loads(doc_to_json(f)))
#        time.sleep(10)
#
#df = pd.DataFrame.from_records(data)
#
#print(df)
#
#df.to_csv('test.csv')

# constants
EMBEDDING_MODEL = "text-embedding-ada-002"

dataset_path = "test.csv"
df = pd.read_csv(dataset_path)

embedding_cache = {}
embedding_cache_path = "cache.pkl"

try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

openai.api_key = api_keys.OPENAI

def embedding_from_string(
    string: str,
    model: str = EMBEDDING_MODEL,
    embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        print("not in cache")
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open('cache2.pkl', "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]


def print_recommendations_from_strings(
    strings: list[str],
    index_of_source_string: int,
    k_nearest_neighbors: int = 3,
    model=EMBEDDING_MODEL,
) -> list[int]:
    """Print out the k nearest neighbors of a given string."""
    # get embeddings for all strings
    embeddings = []

    for string in strings:
        embeddings.append(embedding_from_string(string, model=model))
        #time.sleep(20)
        print('done')

    # get the embedding of the source string
    query_embedding = embeddings[index_of_source_string]
    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
    print(distances)
    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    # print out source string
    query_string = strings[index_of_source_string]
    print(f"Source string: {query_string}")
    # print out its k nearest neighbors
    k_counter = 0
    for i in indices_of_nearest_neighbors:
        # skip any strings that are identical matches to the starting string
        if query_string == strings[i]:
            continue
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1
        # print out the similar strings and their distances
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        String: {strings[i]}
        Distance: {distances[i]:0.3f}"""
        )
    return indices_of_nearest_neighbors, distances

def get_related_data(submission):
    # return data from Luke
    related_data = None
    print(submission)
    d = pd.DataFrame.from_dict(submission, orient='index')
    d = d.transpose()
    print(d)
    df2 = pd.concat([df, d], ignore_index=True)
    print(df2)
    #strings = df['Summary'].to_list()
    #strings.append(submission['Summary'])

    print(df2.iloc[len(df2)-1])

    recc, dist = print_recommendations_from_strings(strings=df2['Summary'].to_list(),
                                                    index_of_source_string=len(df2)-1, k_nearest_neighbors=5)
    out = df2.iloc[recc]
    colours = {
        'Email': 'blue',
        'Submission': 'red',
        'Meeting minutes': 'green',
        'Speech': 'yellow'
    }
    out['Color'] = out['Type'].map(colours)
    out['Relevance'] = [1-i for i in dist]

    out = out.rename(columns={'Title': 'Headline', 'Type': 'Topic'})
    out = out[out["Relevance"] > 0.8]
    out = out[out["Relevance"] != 1.0]
    out['Relevance'] = (out['Relevance']-out['Relevance'].min())/(out['Relevance'].max()-out['Relevance'].min())
    out['Relevance'] = out['Relevance'].apply(lambda x: 0.30 if x < 0.30 else x)
    print(out)
    related_data = out

    if related_data is None:
        related_data = fake_data()
    return related_data

#recc = print_recommendations_from_strings(
#    strings=df['Summary'].to_list(),  # let's base similarity off of the article description
#    index_of_source_string=0,  # let's look at articles similar to the first one about Tony Blair
#    k_nearest_neighbors=5,  # let's look at the 5 most similar articles
#)
