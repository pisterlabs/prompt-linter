import pinecone
import openai
import os
import sys
from datetime import datetime

import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.vectorstores import Pinecone

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import scrapeandindex.sparsevectors as sv
import scrapeandindex.query as query_sql
from dotenv import load_dotenv
load_dotenv()

import psycopg2

# To get streamlit secrets
import streamlit as st

# Open AI API key
if not os.getenv('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')\
# Pinecone API key
if not os.getenv('PINECONE_API_KEY'):
        os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY']
PINECONE_API_KEY= os.getenv('PINECONE_API_KEY')
# Pinecone index name
if not os.getenv('PINECONE_INDEX_NAME'):
        os.environ['PINECONE_INDEX_NAME'] = st.secrets['PINECONE_INDEX_NAME']
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
# Pinecone environment
if not os.getenv('PINECONE_ENV'):
        os.environ['PINECONE_ENV'] = st.secrets['PINECONE_ENV']
PINECONE_ENV = os.getenv('PINECONE_ENV')
# Database URL
if not os.getenv('DATABASE_URL'):
        os.environ['DATABASE_URL'] = st.secrets['DATABASE_URL']
DATABASE_URL = os.getenv('DATABASE_URL')

class InputUpdates:

    def __init__(self):

        self.created_at = None
        # potential_facts is a dictionary of facts, with the key being index and the value being {fact: fact, name: factname, potentialupdate: potentialupdate}
        self.raw_facts = ''
        self.potential_facts = {}

    def add_potential_facts(self, listofpotentialfacts):

        self.created_at = datetime.now()
        for i in range(len(listofpotentialfacts)):
            templistitem = listofpotentialfacts[i]
            templist = templistitem.split(',')
            self.potential_facts[i] = {'fact': ' '.join(templist[1:len(templist)-1]),
                                       'name': templist[-1]}
            
    def add_potential_update(self, index, update_text):
            
        if 'potentialupdate' not in self.potential_facts[index]:
            self.potential_facts[index]['potentialupdate'] = update_text

class LangChainPineconeClient:
    
    def __init__(self, pinecone_key = PINECONE_API_KEY, openai_key = OPENAI_API_KEY, index_name=PINECONE_INDEX_NAME, pinecone_env=PINECONE_ENV, database_url=DATABASE_URL):
        """"
        Initialize LangChainPineconeClient with Pinecone API key and OpenAI API key, plus relevant index name
        Args:
            pinecone_key (str): Pinecone API key
            openai_key (str): OpenAI API key
            index_name (str): name of Pinecone index to use
            pinecone_env (str): Pinecone environment to use
            database_url (str): URL of SQL database to use
        """
        ## Initialize with Pinecone API key and OpenAI API key, plus relevant index name
        pinecone.init(api_key=pinecone_key, environment=pinecone_env)

        index = pinecone.Index(index_name)
        self.index = index

        embed = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=openai_key
        )
        self.embed = embed


        self.llm = ChatOpenAI(
            openai_api_key=openai_key,
            model_name='gpt-3.5-turbo',
            temperature=0.0)
        
        self.llm4 = ChatOpenAI(
            openai_api_key=openai_key,
            model_name='gpt-4',
            temperature=0.0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        
        ## Initialize messages for chat 
        self.messages = [SystemMessage(content="You are a helpful assistant.")]
        self.getfactsmessages = [SystemMessage(content="As an expert proofreader, list any independent objective points, in verbatim, based on the user's need and the user input that need to be verified and aren't explained in the input themselves. Make each list item into - According to the input, <<objective fact about the topic and the topic>>, <<fact name>>")]
        self.getupdatesmessages = [SystemMessage(content="You are a fact checker and help experts keep their information up to date. Understand the user input, and compare it with the latest information attached. If the input is different or out-of-date, provide corrections with an explanation. If the latest information doesn't provide any explanation, say you couldn't find the latest information.")]

        ## Initialize connection to SQL database
        conn = psycopg2.connect(database_url)
        self.cur = conn.cursor()
        self.conn = conn

        self.input_updates = InputUpdates()
        
    def view_indexes(self):
        ## View all indexes
        pinecone.list_indexes()
    
    def get_relevant_text(self, input, topic = "LangChain or prompting LLMs with chains of text"):
        """"
        Initialize LangChainPineconeClient with Pinecone API key and OpenAI API key, plus relevant index name
        Args:
            input (str): input text from user (could be scraped from url)
            topic (str): topic of text to be extracted
        """
        prompt = f'Extract text relevant to {topic} from the following document ' + input
        first_message = HumanMessage(content=prompt)
        self.messages.append(first_message)
        response = self.llm(self.messages)
        self.messages.append(response)
        return response.content
    
    def get_relevant_pinecone_data(self, input):
        """
        Get relevant data from Pinecone index
        Args:
            input (str): extracted text from input or another input 
        Return: 
            (str): relevant text from Pinecone index with source URL
        """
        ##TODO change to return multiple sources to check against rather than just one
        embedded_query = self.embed.embed_query(input)
        query_results = self.index.query(namespace='langchaindocs', top_k=1, \
                                         vector=embedded_query, 
                                         sparse_vector = sv.get_sparse_vector(input))
        match_id = query_results['matches'][0]['id']
        data = query_sql.get_heading_by_rowid(match_id, self.cur)
        result_text = data[5]
        try: #try to get source URL if it exists
            url = query_sql.get_url_by_headingid(match_id, self.conn)
            web_str, verified, date = url #unpack tuple
            result_text = result_text
        except:
            pass
        return result_text, web_str, verified, date

    # update based on the multipayer prompting
    def get_potential_facts(self, input):
        """
        potential facts are independent standalone info points in the user input that can be verified for accuracy

        Args:
            input (str): extracted text from input or another input
        Return:
            (list): list of potential facts
        """
        prompt = f'Need:\nTo see of the info is up to date\n\nInput:\n{input}'
        # Set the initial prompt as the first message
        first_message = HumanMessage(content=prompt)
        # Append the first message to the list of messages
        self.getfactsmessages.append(first_message)
        # Get the response from the LLM in form of a list of facts
        response = self.llm4(self.getfactsmessages)
        # Append the list of facts response to the list of messages
        self.getfactsmessages.append(response)
        # Split the response into a list of facts by each line
        fact_list = response.content.split('\n')
        # Add the list of facts to the InputUpdates object
        self.input_updates.raw_facts = response.content
        self.input_updates.add_potential_facts(fact_list) # list item is in the form '- (item1, fact, factname)'
        return '\n'.join([x['fact'] for x in self.input_updates.potential_facts.values()])
    
    # update based on the multipayer prompting
    def get_potential_updates(self, index):
        """
        potential updates are text that can be used to update the latest information

        Args:
            input (str): extracted text from input or another input
        Return:
            (list): list of potential updates
        """
        fact = self.input_updates.potential_facts[index]['fact']
        data, url, verified, date = self.get_relevant_pinecone_data(fact)
        prompt = f'''Latest information: {data}

        Input: {fact}'''
        context_ask = HumanMessage(content=prompt)
        self.getupdatesmessages.append(context_ask)
        response = self.llm(self.getupdatesmessages)
        self.getupdatesmessages.append(response)
        overall_update = response.content
        self.input_updates.add_potential_update(index, overall_update)
        texttoreturn = f"{self.input_updates.potential_facts[index]['potentialupdate']}\n\nURL: {url}\nVerified: {verified}\nAs of: {date}"
        return texttoreturn
    
    def get_all_updates(self):
        """
        get all updates from the input

        Args:
            input (str): extracted text from input or another input
        Return:
            (list): list of all updates
        """

        # get all potential facts
        self.get_potential_facts()

        # 'potentialupdate' not in self.input_updates.potential_facts[index] get the potential update
        for index in range(len(self.input_updates.potential_facts)):
            if 'potentialupdate' not in self.input_updates.potential_facts[index]:
                self.get_potential_updates(index)

        return self.input_updates.potential_facts
    
    def ask_with_context(self, input, topic = "LangChain or prompting LLMs with chains of text"):
        """
            Calls get_relevant_text and get_relevant_pinecone_data to do the total text flow. 
            This is the only function that needs to be called by the frontend
            Args:
                input (str): extracted text from input or another input 
                topic (str): description of topic of text to be extracted
            Return: 
                (str): relevant text from Pinecone index with source URL
        """
        ##TODO add a function call before this to break things down into subtopics first
        relevant_text = self.get_relevant_text(input, topic)
        data = self.get_relevant_pinecone_data(relevant_text)

        prompt = "Using the following source, verify that the text is up to date and accurate."
        total_prompt = prompt + f' Source: {data}' + f' Text: {relevant_text}'
        context_ask = HumanMessage(content=total_prompt)
        self.messages.append(context_ask)
        response = self.llm(self.messages)
        self.messages.append(response)
        overall_response = response.content + f' Source: {data}'
        return overall_response


    def check_syllabus(self, syllabus):
        pass