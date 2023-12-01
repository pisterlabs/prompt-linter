import cohere

from weaviate.util import generate_uuid5

from langchain.llms import Cohere
from langchain.chat_models import ChatCohere
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings


from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.docstore.document import Document

from unidecode import unidecode

import guardrails as gd
from guardrails.validators import ValidRange, ValidChoices
from pydantic import BaseModel, Field
from rich import print
from typing import List

import PyPDF2
import os
import googlemaps
from datetime import datetime

# google map api: https://github.com/googlemaps/google-maps-services-python --------------------------------


# Utilities ------------------------------
def get_text_from_pdf(fileobj):
    #create reader variable that will read the pdffileobj
    reader = PyPDF2.PdfReader(fileobj)
    
    #This will store the number of pages of this pdf file
    num_pages = len(reader.pages)
    
    combined_text = ''
    for i in range(num_pages):
        # create a variable that will select the selected number of pages
        pageobj = reader.pages[i]
        text = unidecode(pageobj.extract_text())    # remove unnecessary unicode characters
        combined_text += text
        combined_text += '\n'

    return combined_text

# Response  ------------------------------
def get_summary(text):
    '''Return summary using co.summarize endpoint and a two-stage map-reduce approach'''
    
    #split text recursively
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    splits = text_splitter.split_text(text)

    co = cohere.Client(os.environ["COHERE_API_KEY"]) # This is your trial API key

    section_summaries = []
    for t in splits[:]:
        response = co.summarize( 
            text=t,
            length='long',
            format='auto',
            model='command',
            additional_command='focusing on the section summary and details',
            temperature=0,
        )
        section_summaries.append(response.summary)

    combined_section_summaries = '\n\nNew Section Summary \n\n'.join(section_summaries)

    with open('section_summaries', 'w') as f:
        f.write(combined_section_summaries)

    response = co.summarize( 
        text=combined_section_summaries,
        length='long',
        format='auto',
        model='command',
        additional_command='combined the section summaries with focus on client, project scope/description, project location and expected timeline',
        temperature=0,
    )

    class Title(BaseModel):
        title: str = Field(description="Descriptive Title for the RFP summary")

    prompt = '''
        Given the following RFP summary, please extract a dictionary that contains a descriptive title of the project. ONLY output the dictionary and DO NOT ask follow-up questions.

        ${summary}

        ${gr.complete_json_suffix_v2}
    '''

    co = cohere.Client(os.environ["COHERE_API_KEY"]) # This is your trial API key

    guard = gd.Guard.from_pydantic(Title, prompt=prompt)

    raw_llm_output, validated_output = guard(
        co.generate,
        prompt_params={"summary": response.summary},
        model='command',
        temperature=0,
        stop_sequences=["}"],
    )

    print(raw_llm_output)
    
    return response.summary, validated_output['title']

def get_location(summary):
    '''Return location to feed into Google Maps API'''

    class Location(BaseModel):
        intersection: str = Field(description="Major road intersection nearest to the site location. Must contain two road names, the city, and the country")

    prompt = '''
        Given the following RFP summary, please extract extract a dictionary that contains nearest major road intersection to the site location.

        ${summary}

        ${gr.complete_json_suffix_v2}
    '''

    co = cohere.Client(os.environ["COHERE_API_KEY"]) # This is your trial API key

    guard = gd.Guard.from_pydantic(Location, prompt=prompt)

    raw_llm_output, validated_output = guard(
        co.generate,
        prompt_params={"summary": summary},
        model='command',
        temperature=0,
        stop_sequences=["}"],
    )
        
    return validated_output['intersection']

def get_coordinates(intersection):
    '''Get coordinates from Google Maps API'''

    gmaps = googlemaps.Client(key=os.environ["GOOGLE_API_KEY"])
    
    print(intersection)

    # Geocoding an address
    geocode_result = gmaps.geocode(intersection)

    try:
        location = geocode_result[0]['geometry']['location']
        return [c for c in location.values()]
    
    except:
        return [0, 0]

def chat_from_database(prompt: str, chat_history: list=[]) -> str:
    ''' Return response based on the given input '''

    co = cohere.Client(os.environ['COHERE_API_KEY']) # This is your trial API key
    response = co.chat( 
        chat_history=chat_history,
        preamble_override='You are an engineering proposal expert with experience in the infrastructure industry. Support your answer with search as much as possible.',
        message=prompt,
        prompt_truncation='AUTO',
        connectors=[{"id": "weaviate-cfa-proposal-xyt464"}],
        return_chat_history=True
    )

    print("-------------------------------------------")
    # schema: [{'start': str, 'end': str, 'text': str, 'document_ids': str}]
    citations = response.citations
    # schema: [{'filename': str, 'id': str, 'page_number': str, 'text': str}]
    documents = response.documents

    doc_map = {doc['id']: doc for doc in documents}

    # print(response)
    
    # insert citation element in response
    output = ''
    references = {}
    i = 0

    # only if documents were retrieved
    if citations is not None:
        for num, c in enumerate(citations):
            j = c['start']
            k = c['end']

            # print('start', j)
            # print('end', k)

            ids = c['document_ids']

            output += response.text[i:j]
            output += ':blue[{}]'.format(response.text[j:k])

            # print(response.text[j:k])

            # get titles
            titles = [doc_map[id]['title'] for id in ids]
            for t in titles:
                if not (t in references):
                    references[t] = str(len(references) + 1)
            
            # add footnotes ----------------------
            output += ' :grey[*'
            for t in set(titles):
                output += '[{}]'.format(references[t])
            output += '*]'

            # reset start index --------------------
            i = k
        
    output += response.text[i:] # add in t the st

    # add references --------------------
    output += '\n\n'
    for title, i in references.items():
        output += ':grey[[{}] *{}*]  \n'.format(i, title)

    return output, response.text

def get_swot_analysis(summary):

    class SWOT(BaseModel):
        strength: str = Field(description="Company's strength in winning the RFP. Provide references to previous projects.")
        weakness: str = Field(description="Company's weakness in the RFP competition. Provide references to previous projects.")
        opportunities: str = Field(description="Company's opportunities after winning the RFP. Provide references to previous projects.")
        risk: str = Field(description="Company's risk in taking on the project. Provide references to previous projects.")
        decision: str = Field(description="Decide if the company should pursue the project", validators=[ValidChoices(["go", "no go"], on_fail="reask")])

    prompt = f'''
        Given the following RFP summary, conduct a competitiveness analysis (i.e. strength, weakness, opportunity, threat) for this RFP based on the company's project portfolio in the past. 
        Try to provide as many relevant past project examples as possible.

        {summary}
    '''

    return swot_api(prompt)

    # ${gr.complete_json_suffix_v2}

    # guard = gd.Guard.from_pydantic(SWOT, prompt=prompt)

    # raw_llm_output, validated_output = guard(
    #     swot_api,
    #     prompt_params={"summary": summary},
    # )

    # print(raw_llm_output)

    # return raw_llm_output
    

def swot_api(prompt: str, **kwargs) -> str:
    """Custom LLM API wrapper.

    Args:
        prompt (str): The prompt to be passed to the LLM API
        **kwargs: Any additional arguments to be passed to the LLM API

    Returns:
        str: The output of the LLM API
    """

    co = cohere.Client(os.environ["COHERE_API_KEY"])

    response = co.chat( 
        preamble_override='You are an engineering proposal expert with experience in the infrastructure industry. Support your answer with search as much as possible.',
        message=prompt,
        prompt_truncation='AUTO',
        connectors=[{"id": "weaviate-cfa-proposal-xyt464"}],
        return_chat_history=True,
    )

    # Call your LLM API here
    return response.text
