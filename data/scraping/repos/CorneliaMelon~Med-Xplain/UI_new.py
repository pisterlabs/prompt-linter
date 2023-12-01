from urllib.parse import urljoin

import dotenv
import pinecone
import requests
import streamlit as st
from humanloop import Humanloop
from langchain.chains import ConversationChain
from langchain.chains import create_extraction_chain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from webscraper import call_nhs_search
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
# from utils import *
from sentence_transformers import SentenceTransformer

dotenv.load_dotenv()
import openai

import urllib.parse

dotenv.load_dotenv()
import os
import asyncio
import dotenv

dotenv.load_dotenv()
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
OPENAI_API = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API
cache_folder = "path/to/your/cache/directory"
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_folder)

pinecone.init(
    api_key="0deabe30-31e5-4385-a193-bbeff060b252",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
)
index = pinecone.Index('chatbot')

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)
# Constants

HUMAN_LOOP_API_KEY = os.getenv("HUMAN_LOOP_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
#  # add project ID
# PROJECT_ID = 'pr_KhtlyvJhwaWe0AE53pFLe'
hl = Humanloop(api_key=HUMAN_LOOP_API_KEY)

template_text = """
        - Your primary goal is to provide information and guidance about medical treatment options based on the user's questions. Below is an example question; please answer accordingly.Don't include words like i am not a doctor,i am ai bot
        - Question: 'What are the treatment options for asthma?'

        Example:
        - Question: 'Can you explain the different treatment options for asthma?'
        - Answer: 'Certainly! There are several treatment options for asthma, including:
          1. Inhalers (e.g., bronchodilators and corticosteroid inhalers) to relieve symptoms and reduce inflammation.
          2. Corticosteroids (oral or inhaled) to control inflammation in the airways.
          3. Immunotherapy for individuals with severe allergic asthma.
          
          Please note that the specific treatment plan for asthma can vary depending on the severity of the condition and individual patient needs. Med-Xplain is an assistive technology. Please consult your physician for further guidance and prescriptions.'
        """
system_msg_template = SystemMessagePromptTemplate.from_template(template=template_text)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API)
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


def search_papers(search_term, page=1):
    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    url = f'{base_url}?db=pubmed&term={search_term}&retmode=json&retstart={((page - 1) * 3)}&retmax=3'
    try:
        response = requests.get(url)
        data = response.json()
        pubmed_ids = data['esearchresult']['idlist']

        results = []

        for pubmed_id in pubmed_ids:
            summary_url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pubmed_id}&retmode=json'
            summary_response = requests.get(summary_url)
            summary_data = summary_response.json()
            article_title = summary_data['result'][pubmed_id]['title']
            article_url = f'https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/'
            authors = summary_data['result'][pubmed_id]['authors']
            author_names = [author['name'] for author in authors]
            final_response = f"<a href='{article_url}' target='_blank'>{article_title}</a> by {', '.join(author_names)}"
            results.append(final_response)

        for result in results:
            print(result)

        return results

    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    # return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']
    return result['matches'][0]


def query_refiner(query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query, refine it to be most suitable for retrieving an answer from a knowledge base:\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text'].strip()


def extract(content: str, schema: dict):
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",
                     openai_api_key=OPENAI_API)

    return create_extraction_chain(schema=schema, llm=llm).run(content)


def call_nhs_search(query):
    from langchain.document_loaders import AsyncChromiumLoader
    # from langchain.document_transformers import BeautifulSoupTransformer

    from bs4 import BeautifulSoup
    # Load HTML
    loader = AsyncChromiumLoader(
        [f"https://www.nhs.uk/search/results?q={urllib.parse.quote(query)})"])
    html = loader.load()

    soup = BeautifulSoup(html[0].page_content, "html.parser")
    link_tags = soup.find_all('a')

    # Create a list of dictionaries with title and href as key-value pairs
    links_with_titles = [
        {"page_title": tag.get_text().strip("\n "), "page_url": "https://www.nhs.uk/" + tag.get('href'), } for
        tag in link_tags[11:-10]]

    return links_with_titles


def run_conversation(content):
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": content}]
    response = hl.chat_deployed(
        project_id=PROJECT_ID,
        messages=messages,
    )

    response = response.body["data"][0]  # first response

    if response.get("output") != None:
        # Step 2: call the function
        tool_name = response["output"]
        # TODO: pubMed needs to return search arguments

        if tool_name.startswith('pubMed'):

            pubmed_args = response["output"].split("-")[1:]
            tool_result = search_papers(search_term=pubmed_args)
        elif tool_name.startswith('nhs'):
            print("selected nhs")
            # tool_result = "query_wolfram_alpha(query=tool_args.get('query'))"
            nhs_args = response["output"].split("-")[1:]
            tool_result = (
                    "Highlight the following links for my treatment options from the nhs, and also note that they are not a substitute for a physician's advice\t" + str(
                call_nhs_search(query=" ".join(nhs_args))))
            # return tool_result,tool_name
        elif tool_name.startswith('pdf'):
            print("selected pdf")
            conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm,
                                             verbose=True)
            refined_query = query_refiner(response)
            context = find_match(refined_query)
            tool_result = conversation.predict(input=f"Query:\n{context}")
            # tool_result = conversation.predict(input=f"Query:\n{response}")
            # print(tool_result)
            return tool_result
            # tool_result = "query_wolfram_alpha(query=tool_args.get('query'))"

            # response = conversation.predict(input=f"Query:\n{query}")
            # response = limit_words(response)
        else:
            raise NotImplementedError("My code does not know about this tool!")

        # Step 3: send the response back to the model
        messages.append(
            {
                "role": "assistant",
                # "name": tool_name,
                "content": str(tool_result),
            }
        )

        second_response = hl.chat_deployed(
            project_id=PROJECT_ID,
            messages=messages,
        )
        # print(second_response.body["data"])
        return second_response.body["data"][0]["output"]
    else:
        return response


# Streamlit app - run if you want to use Streamlit

def run_app():
    st.title("Med-Xplain")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What can I help with?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call your run_conversation function
        response_data = run_conversation(prompt)

        # if tool_name.startswith('nhs'):
        # Extract the desired message from the response_data
        # response_data = response_data["output"] if response_data else f"Error processing request: {prompt}"

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response_data)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_data})


# Actually run the app
if __name__ == "__main__":
    run_app()
